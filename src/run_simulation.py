import numpy as np
import os
from .fishery_simulation import FisherySimulation
from .data_logger import DataLogger
from .config import SIMULATION_PARAMS, INTENSE_PARAMS, RESULTS_FOLDER

def run_simulation(config, target_w=INTENSE_PARAMS["target_w_range"].mean(), random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)  # set random seed for reproducibility

    results_folder = config.get("results_folder", RESULTS_FOLDER)
    os.makedirs(results_folder, exist_ok=True)  # ensure results folder exists

    sim_params = config["sim_params"]  # extract simulation parameters

    # Use burn_in_steps from SIMULATION_PARAMS
    burn_in_steps = sim_params.get("burn_in_steps", SIMULATION_PARAMS["burn_in_steps"])
    # Use recruitment_month and num_survey_stations from INTENSE_PARAMS
    recruitment_month = INTENSE_PARAMS["recruitment_month"]
    num_survey_stations = INTENSE_PARAMS["num_survey_stations"]

    simulation = FisherySimulation(
        initial_abundance=sim_params["initial_abundance"],  # starting abundance
        fishing_effort=sim_params["initial_effort"],  # starting effort
        num_patches=INTENSE_PARAMS["num_patches"],  # number of patches
        patch_sizes=INTENSE_PARAMS["patch_sizes"],  # patch sizes
        patch_open=INTENSE_PARAMS["patch_open"],  # which patches are open to fishing
        age_classes=INTENSE_PARAMS["age_classes"],  # number of age classes
        bh_alpha=INTENSE_PARAMS["bh_alpha"],  # Beverton-Holt alpha
        bh_beta=INTENSE_PARAMS["bh_beta"],  # Beverton-Holt beta
        selectivity=INTENSE_PARAMS["selectivity"],  # age-specific selectivity
        catchability=INTENSE_PARAMS["catchability"],  # catchability coefficient
        natural_mortality=INTENSE_PARAMS["natural_mortality"],  # natural mortality
        weight_at_age=INTENSE_PARAMS["weight_at_age"],  # weight per age class
        maturity_at_age=INTENSE_PARAMS["maturity_at_age"],  # maturity per age
        seasonal_factors=INTENSE_PARAMS.get("seasonal_factors", {}),  # seasonal effects
        env_noise=INTENSE_PARAMS["env_noise"],  # environmental stochasticity
        survey_error_sd=INTENSE_PARAMS["survey_error_sd"],  # survey observation error
        recruitment_scenario=INTENSE_PARAMS["recruitment_scenario"],  # recruitment type
        alpha_final=INTENSE_PARAMS["alpha_final"],  # final alpha for recruitment
        drop_year=INTENSE_PARAMS["drop_year"],  # year recruitment drops
        max_years=sim_params["time_steps"],  # total simulation steps
        target_abundance=sim_params["target_abundance"],  # target abundance
        burn_in_steps=burn_in_steps,  # burn-in period from SIMULATION_PARAMS
        target_w=target_w,  # control parameter w
        manager_params={"w": target_w}  # pass w to manager
    )

    logger = DataLogger()  # initialize data logger

    simulation.burn_in()  # run burn-in steps to stabilize population

    for t in range(sim_params["time_steps"]):
        diag = simulation.step()  # advance one timestep
        logger.log(
            t,
            diag["total_abundance"],  # true abundance
            diag["observed_abundance"],  # manager sees this
            diag["fishing_effort"],  # applied effort
            diag.get("catch_biomass")  # optional catch
        )

    return logger.summarize()  # return results as a DataFrame
