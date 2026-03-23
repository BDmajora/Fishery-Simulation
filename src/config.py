import numpy as np

INTENSE_PARAMS = {
    "num_patches": 1,  # number of patches/areas in the simulation
    "patch_sizes": (26067,),  # size of each patch
    "patch_open": (True,),  # whether fishing is allowed in each patch
    "age_classes": 6,  # number of age classes
    "selectivity": np.array([0.45, 1.0, 1.0, 1.0, 1.0, 1.0]),  # fishing selectivity by age
    "weight_at_age": np.array([10.08, 29.66, 52.67, 74.21, 92.18, 106.07]),  # weight of fish by age
    "maturity_at_age": np.array([0.37, 0.74, 0.90, 0.95, 0.97, 1.0]),  # proportion mature by age
    "natural_mortality": 0.1,  # natural mortality rate
    "catchability": 0.5,  # catchability coefficient
    "bh_alpha": 0.1427,  # Beverton-Holt alpha parameter
    "bh_beta": 2.56e-9,  # Beverton-Holt beta parameter
    "env_noise": 0.1,  # environmental noise standard deviation
    "survey_error_sd": 0.2,  # survey observation error (log-normal)
    "num_survey_stations": 18,  # number of survey stations (configurable)
    "recruitment_month": 0,  # month recruitment occurs (configurable)
    "seasonal_factors": {4: 1.5, 10: 0.8},  # recruitment scaling for May and Nov
    "seasonal_effort": np.array([  # fraction of annual effort per month
        0.05, 0.05, 0.05, 0.07, 0.10, 0.10, 0.08, 0.08, 0.10, 0.12, 0.15, 0.05
    ]) / np.sum(np.array([  # normalize to sum to 1
        0.05, 0.05, 0.05, 0.07, 0.10, 0.10, 0.08, 0.08, 0.10, 0.12, 0.15, 0.05
    ])),
    "recruitment_scenario": "stationary",  # type of recruitment scenario
    "alpha_final": 0.07135,  # final alpha value for recruitment adjustment
    "drop_year": 20,  # year when recruitment drops
    "target_w_range": np.arange(0.1, 1.0 + 0.0001, 0.1)  # range of target control parameters
}

SIMULATION_PARAMS = {
    "initial_abundance": 5000,  # starting population
    "initial_effort": 0.1,  # starting fishing effort
    "time_steps": 100,  # number of simulation steps
    "target_abundance": 5000,  # target stock level
    "burn_in_steps": 12  # initial steps ignored for stabilization
}

RESULTS_FOLDER = "results"  # folder to save simulation outputs

HIGH_HISTORICAL_EFFORT = 0.5  # high effort scenario
LOW_HISTORICAL_EFFORT = 0.05  # low effort scenario

RECRUITMENT_SCENARIOS = ["stationary", "sudden", "gradual"]  # types of recruitment scenarios
ENV_NOISE_RANGE = [0.0, 0.1, 0.2]  # range of environmental noise

SEASONAL_VARIATIONS = [  # different seasonal factor variations
    {4: 1.5, 10: 0.8},
    {4: 2.6133, 10: 0.5351},
]

PATCH_STRUCTURES = [  # different patch configurations
    {"patch_sizes": (0.9566, 0.0434), "patch_open": (True, False)},
    {"patch_sizes": (0.7, 0.3), "patch_open": (True, True)},
]
