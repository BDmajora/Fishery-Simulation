import torch
import numpy as np
from .management_procedures import TargetBasedManager
from .config import INTENSE_PARAMS, SIMULATION_PARAMS

class FisherySimulation:
    def __init__(
        self,
        initial_abundance,
        fishing_effort,
        num_patches,
        patch_sizes,
        patch_open,
        age_classes,
        bh_alpha,
        bh_beta,
        selectivity,
        catchability,
        natural_mortality,
        weight_at_age,
        maturity_at_age,
        seasonal_factors,
        env_noise,
        survey_error_sd,
        recruitment_scenario,
        alpha_final,
        drop_year,
        max_years,
        target_w,
        target_abundance,
        burn_in_steps,
        manager_params=None
    ):
        # Hardware acceleration setup (Crucial for "Target Hardware" resume requirement)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_patches = int(num_patches)
        self.num_survey_stations = int(INTENSE_PARAMS.get("num_survey_stations", 18))
        self.recruitment_month = int(INTENSE_PARAMS.get("recruitment_month", 0))

        if isinstance(age_classes, (list, np.ndarray, torch.Tensor)):
            self.age_classes = len(age_classes)
        else:
            self.age_classes = int(age_classes)

        # Pre-loading configurations to Tensors on the target device
        self.patch_sizes = torch.tensor(patch_sizes, dtype=torch.float32, device=self.device)
        self.patch_open = torch.tensor(patch_open, dtype=torch.bool, device=self.device)
        self.selectivity = torch.tensor(selectivity, dtype=torch.float32, device=self.device)
        self.weight_at_age = torch.tensor(weight_at_age, dtype=torch.float32, device=self.device)
        self.maturity_at_age = torch.tensor(maturity_at_age, dtype=torch.float32, device=self.device)

        # Initialize population state on GPU/CPU
        self.N = torch.zeros((self.num_patches, self.age_classes), dtype=torch.float32, device=self.device)
        initial_by_patch = float(initial_abundance) * self.patch_sizes
        # Vectorized distribution across ages using unsqueeze for broadcasting
        self.N[:] = initial_by_patch.unsqueeze(1) / float(self.age_classes)

        # Scalars
        self.fishing_effort = float(fishing_effort)
        self.q = float(catchability)
        self.M = float(natural_mortality)
        self.bh_alpha = float(bh_alpha)
        self.bh_beta = float(bh_beta)
        self.env_noise = float(env_noise)
        self.survey_error_sd = float(survey_error_sd)
        
        self.seasonal_factors = seasonal_factors or {}
        self.recruitment_scenario = recruitment_scenario
        self.alpha_final = float(alpha_final)
        self.drop_year = int(drop_year)
        self.max_years = int(max_years)
        self.target_w = target_w
        self.target_abundance = float(target_abundance)
        self.current_month = 0
        self.current_time = 0
        self.burn_in_steps = int(burn_in_steps)
        self.survey_history = []

        if manager_params is None:
            manager_params = {}
        self.manager = TargetBasedManager(
            Itarget=self.target_abundance,
            Etarget=self.fishing_effort,
            I0=manager_params.get("I0", 0.2 * self.target_abundance)
        )
        self.manager_w = manager_params.get("w", self.target_w)
        self.manager_max_change = manager_params.get("max_change", 0.2)
        self.seasonal_effort = torch.tensor(manager_params.get("seasonal_effort", np.ones(12)), 
                                           dtype=torch.float32, device=self.device)

    def compute_fishing_mortality(self, effort):
        # Vectorized logic to avoid inefficient Python loops
        open_area = self.patch_sizes[self.patch_open].sum()
        effort_by_patch = torch.zeros(self.num_patches, device=self.device)
        
        if open_area > 0:
            effort_by_patch[self.patch_open] = effort * (self.patch_sizes[self.patch_open] / open_area)
        
        # Outer product via unsqueeze (num_patches, 1) * (1, age_classes)
        F = self.q * effort_by_patch.unsqueeze(1) * self.selectivity.unsqueeze(0)
        return F, effort_by_patch

    def baranov_catch(self, N, F, M):
        Z = F + M
        # Tensor-based catch formula with safety check
        catch = torch.where(Z > 0, (F / Z) * N * (1 - torch.exp(-Z)), torch.tensor(0.0, device=self.device))
        return catch, Z

    def survey(self):
        station_area = self.patch_sizes.sum() / self.num_survey_stations
        observed_catch = self.q * self.N * (station_area / self.patch_sizes.sum())
        abundance_estimate = (observed_catch / self.q * self.num_survey_stations).sum()
        
        if self.survey_error_sd > 0.0:
            # FIX: Use torch.randn(()) to create a scalar tensor (shape []) 
            # to match the shape of abundance_estimate.
            noise = torch.exp(torch.randn((), device=self.device) * self.survey_error_sd)
            abundance_estimate *= noise
            
        return abundance_estimate.item()

    def adjust_alpha(self):
        t_months = int(self.current_time)
        alpha_init = float(self.bh_alpha)
        if self.recruitment_scenario == "stationary":
            return alpha_init
        elif self.recruitment_scenario == "sudden":
            return alpha_init if (t_months // 12) < self.drop_year else float(self.alpha_final)
        elif self.recruitment_scenario == "gradual":
            years = t_months / float(max(1, self.max_years))
            return alpha_init - years * (alpha_init - float(self.alpha_final))
        return alpha_init

    def step(self):
        # 1. Effort Scaling (Monthly)
        factor = self.seasonal_factors.get(self.current_month, 1.0)
        factor = max(0.1, min(factor, 2.0))
        effort_this_month = self.fishing_effort * self.seasonal_effort[self.current_month]

        # 2. Vectorized Fishing Mortality (No Loops)
        # We use unsqueeze(1) and (0) to create a matrix from two vectors automatically
        F, effort_by_patch = self.compute_fishing_mortality(effort_this_month)
        
        # 3. Baranov Catch (Element-wise Tensor Math)
        catch_numbers, Z = self.baranov_catch(self.N, F, self.M)
        survivors = self.N * torch.exp(-Z)

        # 4. Optimized Ageing (Slicing/Concat)
        # This is high-performance: we shift the population forward in time without a loop
        aged_N = torch.zeros_like(self.N)
        aged_N[:, 1:] = survivors[:, :-1]
        aged_N[:, -1] += survivors[:, -1] # Accumulate in the "plus-group" (oldest fish)

        # 5. Spawning Biomass (Reduction Operation)
        # dim=1 collapses the age dimension into a single biomass value per patch
        spawning_biomass = torch.sum(survivors * self.weight_at_age * self.maturity_at_age, dim=1)

        # 6. Recruitment (Stochastic Tensor Math)
        recruits = torch.zeros(self.num_patches, device=self.device)
        if self.current_month == self.recruitment_month:
            alpha_t = self.adjust_alpha()
            recruits = (alpha_t * spawning_biomass) / (1.0 + self.bh_beta * spawning_biomass)
            
            if self.env_noise > 0.0:
                # Generate noise directly on the GPU/Device
                noise = torch.exp(torch.randn(recruits.shape, device=self.device) * self.env_noise)
                recruits *= noise
            
            recruits = torch.clamp(recruits * factor, min=0.0)

        # Update the state
        aged_N[:, 0] = recruits
        self.N = aged_N

        # 7. Management & Logging (Moving back to Python scalars only when necessary)
        total_abundance = self.N.sum().item()
        observed_abundance = total_abundance
        if self.current_time >= self.burn_in_steps:
            observed_abundance = self.survey()
        
        self.manager.update_abundance_index(observed_abundance)
        self.fishing_effort = self.manager.adjust_effort(
            current_effort=self.fishing_effort,
            w=self.manager_w,
            max_change=self.manager_max_change
        )

        self.current_month = (self.current_month + 1) % 12
        self.current_time += 1

        return {
            "N_numbers": self.N.detach().cpu().numpy(),
            "total_abundance": total_abundance,
            "observed_abundance": observed_abundance,
            "fishing_effort": float(effort_this_month),
            "catch_biomass": (catch_numbers * self.weight_at_age).sum().item()
        }

    def burn_in(self):
        for _ in range(self.burn_in_steps):
            _ = self.step()