import numpy as np
from .management_procedures import TargetBasedManager
from .config import INTENSE_PARAMS, SIMULATION_PARAMS

class FisherySimulation:
    def __init__(
        self,
        initial_abundance,  # total starting abundance for the fishery
        fishing_effort,  # initial fishing effort
        num_patches,  # number of spatial patches
        patch_sizes,  # size of each patch
        patch_open,  # which patches are open to fishing
        age_classes,  # number of age classes or array of ages
        bh_alpha,  # Beverton-Holt alpha parameter for recruitment
        bh_beta,  # Beverton-Holt beta parameter for recruitment
        selectivity,  # selectivity at age for fishing
        catchability,  # catchability coefficient
        natural_mortality,  # natural mortality rate
        weight_at_age,  # weight of fish by age
        maturity_at_age,  # maturity proportion by age
        seasonal_factors,  # seasonal recruitment or fishing adjustment
        env_noise,  # environmental stochasticity
        survey_error_sd,  # observation error in survey
        recruitment_scenario,  # type of recruitment scenario
        alpha_final,  # final alpha value for changing recruitment
        drop_year,  # year when recruitment drops in sudden scenario
        max_years,  # maximum simulation years
        target_w,  # target-based management parameter
        target_abundance,  # target abundance for manager
        burn_in_steps,  # number of burn-in timesteps
        manager_params=None  # additional parameters for the manager
    ):
        # number of patches
        self.num_patches = int(num_patches)
        # number of survey stations, read from INTENSE_PARAMS
        self.num_survey_stations = int(INTENSE_PARAMS.get("num_survey_stations", 18))
        # month when recruitment occurs, read from INTENSE_PARAMS
        self.recruitment_month = int(INTENSE_PARAMS.get("recruitment_month", 0))

        # determine number of age classes
        if isinstance(age_classes, (list, np.ndarray)):
            self.age_classes = len(age_classes)
        else:
            self.age_classes = int(age_classes)

        # patch sizes and open status
        self.patch_sizes = np.array(patch_sizes, dtype=float)
        self.patch_open = np.array(patch_open, dtype=bool)

        # population numbers by patch and age
        self.N = np.zeros((self.num_patches, self.age_classes), dtype=float)
        initial_by_patch = float(initial_abundance) * self.patch_sizes
        for a in range(self.age_classes):
            self.N[:, a] = initial_by_patch / float(self.age_classes)  # distribute initial abundance evenly

        # fishing parameters
        self.fishing_effort = float(fishing_effort)
        self.q = float(catchability)  # catchability coefficient
        self.M = float(natural_mortality)  # natural mortality
        self.selectivity = np.asarray(selectivity, dtype=float)  # selectivity at age
        if self.selectivity.shape[0] != self.age_classes:
            raise ValueError("selectivity length must equal age_classes")

        # Beverton-Holt parameters and fish properties
        self.bh_alpha = float(bh_alpha)
        self.bh_beta = float(bh_beta)
        self.weight_at_age = np.asarray(weight_at_age, dtype=float)
        self.maturity_at_age = np.asarray(maturity_at_age, dtype=float)

        # seasonal factors and stochasticity
        self.seasonal_factors = seasonal_factors or {}  # seasonal recruitment/fishing factors
        self.env_noise = float(env_noise)  # environmental noise
        self.survey_error_sd = float(survey_error_sd)  # survey observation error

        # recruitment scenario and parameters
        self.recruitment_scenario = recruitment_scenario
        self.alpha_final = float(alpha_final)
        self.drop_year = int(drop_year)
        self.max_years = int(max_years)

        # target-based management
        self.target_w = target_w
        self.target_abundance = float(target_abundance)

        # bookkeeping
        self.current_month = 0  # current simulation month
        self.current_time = 0  # current timestep
        self.burn_in_steps = int(burn_in_steps)  # burn-in steps
        self.survey_history = []  # record of observed abundance

        # manager parameters
        if manager_params is None:
            manager_params = {}
        self.manager = TargetBasedManager(
            Itarget=self.target_abundance,
            Etarget=self.fishing_effort,
            I0=manager_params.get("I0", 0.2 * self.target_abundance)
        )
        self.manager_w = manager_params.get("w", self.target_w)  # control parameter w
        self.manager_max_change = manager_params.get("max_change", 0.2)  # max change in effort per step

        # monthly effort allocation
        self.seasonal_effort = manager_params.get("seasonal_effort", np.ones(12))
        if len(self.seasonal_effort) != 12:
            raise ValueError("seasonal_effort must be length 12 (months)")

    def compute_fishing_mortality(self, effort):
        open_area = self.patch_sizes[self.patch_open].sum()  # total area open to fishing
        effort_by_patch = np.zeros(self.num_patches, dtype=float)  # distribute effort by patch
        if open_area > 0:
            effort_by_patch[self.patch_open] = effort * (self.patch_sizes[self.patch_open] / open_area)
        F = np.zeros((self.num_patches, self.age_classes), dtype=float)  # fishing mortality matrix
        for i in range(self.num_patches):
            F[i, :] = self.q * effort_by_patch[i] * self.selectivity  # age-specific mortality
        return F, effort_by_patch

    def baranov_catch(self, N, F, M):
        Z = F + M  # total mortality
        with np.errstate(divide='ignore', invalid='ignore'):
            catch = np.where(Z > 0, (F / Z) * N * (1 - np.exp(-Z)), 0.0)  # Baranov catch formula
        return catch, Z

    def survey(self):
        station_area = self.patch_sizes.sum() / self.num_survey_stations  # area per station
        observed_catch = self.q * self.N * (station_area / self.patch_sizes.sum())  # simulated survey catch
        abundance_estimate = observed_catch / self.q * self.num_survey_stations  # estimate total abundance
        total_estimate = float(np.sum(abundance_estimate))  # sum across patches and ages
        if self.survey_error_sd > 0.0:
            total_estimate *= np.exp(np.random.normal(0.0, self.survey_error_sd))  # log-normal observation error
        return total_estimate

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
        factor = self.seasonal_factors.get(self.current_month, 1.0)  # seasonal recruitment factor
        factor = max(0.1, min(factor, 2.0))  # clamp factor
        effort_this_month = self.fishing_effort * self.seasonal_effort[self.current_month]  # scale effort

        F, effort_by_patch = self.compute_fishing_mortality(effort_this_month)  # fishing mortality
        catch_numbers, Z = self.baranov_catch(self.N, F, self.M)  # Baranov catch
        survivors = self.N * np.exp(-Z)  # surviving numbers after mortality

        aged_N = np.zeros_like(self.N)
        aged_N[:, 1:] = survivors[:, :-1]  # age the population
        aged_N[:, -1] += survivors[:, -1]  # plus oldest age class

        spawning_biomass = np.sum(survivors * self.weight_at_age[None, :] * self.maturity_at_age[None, :], axis=1)  # spawning biomass

        recruits = np.zeros(self.num_patches, dtype=float)
        if self.current_month == self.recruitment_month:
            alpha_t = self.adjust_alpha()  # get recruitment parameter
            recruits = (alpha_t * spawning_biomass) / (1.0 + self.bh_beta * spawning_biomass)  # Beverton-Holt recruitment
            if self.env_noise > 0.0:
                recruits *= np.exp(np.random.normal(0.0, self.env_noise, size=recruits.shape))  # add environmental noise
            recruits *= factor  # seasonal adjustment
            recruits = np.maximum(recruits, 0.0)  # prevent negative recruits

        aged_N[:, 0] = recruits  # set new recruits
        self.N = aged_N

        total_abundance = float(self.N.sum())  # total true abundance
        observed_abundance = total_abundance
        if self.current_time >= self.burn_in_steps:
            observed_abundance = self.survey()  # apply survey after burn-in
        self.survey_history.append(observed_abundance)

        self.manager.update_abundance_index(observed_abundance)  # update manager
        self.fishing_effort = self.manager.adjust_effort(
            current_effort=self.fishing_effort,
            w=self.manager_w,
            max_change=self.manager_max_change
        )  # manager adjusts effort

        self.current_month = (self.current_month + 1) % 12  # advance month
        self.current_time += 1  # advance timestep

        catch_biomass_per_age = catch_numbers * self.weight_at_age[None, :]
        total_catch_biomass = float(np.sum(catch_biomass_per_age))  # total biomass caught

        return {
            "N_numbers": self.N.copy(),
            "catch_numbers": catch_numbers,
            "catch_biomass": total_catch_biomass,
            "effort_by_patch": effort_by_patch,
            "Z": Z,
            "F": F,
            "M": self.M,
            "total_abundance": total_abundance,
            "observed_abundance": observed_abundance,
            "total_biomass": float(np.sum(self.N * self.weight_at_age[None, :])),
            "fishing_effort": float(effort_this_month)
        }

    def burn_in(self):
        for _ in range(self.burn_in_steps):
            _ = self.step()  # run burn-in steps without recording
