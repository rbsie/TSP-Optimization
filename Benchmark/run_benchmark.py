import pandas as pd
import numpy as np
import random
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tsp_streamlit import (
    run_tsp_gurobi_mtz,
    run_tsp_gurobi_dfj,
    run_tsp_gurobi_dfj_lazy,
    run_tsp_gurobi_fb
)

# Ensure Reproducibility
np.random.seed(42)
random.seed(42)

# Create folder to save cities
instance_dir = "instances"
os.makedirs(instance_dir, exist_ok=True)

# Load Dataset
df_cities = pd.read_csv("Datasets/top_cities_coordinates_df.csv")

# Define Instance Sizes
sizes = [10, 15, 20, 30, 50, 100, 300, 500]

# Define Formulations
formulations = {
    "MTZ": run_tsp_gurobi_mtz,
    "DFJ_Standard": run_tsp_gurobi_dfj, # standard version
    "DFJ_Lazy": run_tsp_gurobi_dfj_lazy, # lazy version
    "Flow-Based": run_tsp_gurobi_fb
}

# Set Timeout to 30 minutes
timeout = 1800

# Define number of repeats for each instance
repeats = 5

# Run benchmark

# Save all results in csv
# If there is already a file, we will only do the missing instances
results_path = "Benchmark/benchmark_results.csv"
if os.path.exists(results_path):
    df_done = pd.read_csv(results_path)
else:
    df_done = pd.DataFrame(columns=["formulation", "n_cities", "instance",
                                    "runtime", "gap in %", "obj"])

for formulation_name, solver in formulations.items():
    for size in sizes:

        for rep in range(1, repeats + 1):

            if formulation_name == "DFJ_Standard" and size > 20:
                print(f"Skipping DFJ Standard for size {size} (too large)")
                continue

            # Skip combinations already completed
            if ((df_done["formulation"] == formulation_name) &
                (df_done["n_cities"] == size) &
                (df_done["instance"] == rep)).any():

                print(f"Skipping already completed: {formulation_name}, n={size}, rep={rep}")
                continue

            # Create random samples for each repeat
            # random_state=rep will ensure that the same instance is created again for same rep
            sample = df_cities.sample(n=size, random_state=rep).reset_index(drop=True)

            # Save those instances
            path = f"{instance_dir}/cities_{formulation_name}_{size}_{rep}.csv"
            sample.to_csv(path, index=False)

            # Run solver
            print()
            print(f"Now running: formulation {formulation_name}, size: {size}, repetition: {rep}")
            print()
            res = solver(sample, 0, timeout) 

            # Save results (Nan if Gurobi found no feasible solution)
            result_row = {
                "formulation": formulation_name,
                "n_cities": size,
                "instance": rep,
                "runtime": timeout if res["obj"] is None else res["solve_time"],
                "gap in %": np.nan if res["obj"] is None else res["gap"] * 100,
                "obj": np.nan if res["obj"] is None else res["obj"]
            }
            print("Result Row:", result_row)

            # append to df_done
            df_done = pd.concat([df_done, pd.DataFrame([result_row])], ignore_index=True)

            # update df_done
            df_done.to_csv(results_path, index=False)

print('Benchmarking completed! Results saved to Benchmark/benchmark_results.csv')