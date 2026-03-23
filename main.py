from src.config import SIMULATION_PARAMS, RESULTS_FOLDER
from src.run_simulation import run_simulation
from src.plotting import plot_results
from src.saving import save_csv
import os

def run_single_simulation():
    target_w = 0.5  # target value for the management procedure

    # Ask user for number of months to simulate
    user_input = input(f"Enter number of months to simulate [default {SIMULATION_PARAMS['time_steps']} months]: ").strip()
    if user_input:  # if user entered something, use that value
        try:
            months = int(user_input)
            sim_params = SIMULATION_PARAMS.copy()
            sim_params["time_steps"] = months  # set simulation length in months
        except ValueError:
            print("Invalid input, using default simulation length.")
            sim_params = SIMULATION_PARAMS
    else:
        sim_params = SIMULATION_PARAMS  # use default if no input

    # run the simulation with the given parameters and target
    df = run_simulation({"sim_params": sim_params}, target_w=target_w)

    csv_file = "simulation_w0.5.csv"  # name for CSV output
    plot_file = "simulation_w0.5.png"  # name for plot output

    os.makedirs(RESULTS_FOLDER, exist_ok=True)  # make sure results folder exists

    save_csv(df, RESULTS_FOLDER, csv_file)  # save simulation results to CSV
    plot_results(
        df,
        target_abundance=sim_params["target_abundance"],  # used for reference line in plot
        folder=RESULTS_FOLDER,
        filename=plot_file
    )  # save plot of simulation results

    print(f"Results saved: CSV='{csv_file}', Plot='{plot_file}'")  # confirm files saved

if __name__ == "__main__":
    run_single_simulation()  # run the simulation when the script is executed
