import pandas as pd
import numpy as np

class DataLogger:
    def __init__(self):
        self.records = []  # list to store all logged timesteps
        self.initial_abundance = None  # store abundance at first timestep for collapse threshold

    def log(self, time, abundance, observed_abundance, effort, catch_biomass=None):
        if self.initial_abundance is None:
            self.initial_abundance = abundance  # set initial abundance at first log

        collapse_threshold = 0.2 * self.initial_abundance  # threshold for collapse detection
        collapsed = abundance < collapse_threshold  # check if current abundance is below threshold

        record = {
            "Time": time,  # current timestep
            "True_Abundance": abundance,  # actual total abundance
            "Observed_Abundance": observed_abundance,  # survey index used for management
            "Effort": effort,  # fishing effort applied
            "Collapsed": collapsed  # whether population is collapsed
        }

        if catch_biomass is not None:
            # sum catch if it is an array or list
            record["Catch_Biomass"] = np.sum(catch_biomass) if isinstance(catch_biomass, (list, np.ndarray)) else catch_biomass

        self.records.append(record)  # save the record

    def summarize(self):
        return pd.DataFrame(self.records)  # convert all logged records to a pandas DataFrame
