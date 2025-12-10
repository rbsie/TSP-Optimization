import pandas as pd
import numpy as np
import random
import os
from tsp_streamlit import (
    run_tsp_gurobi_mtz,
    run_tsp_gurobi_dfj,
    run_tsp_gurobi_fb
)

# Ensure Reproducibility
np.random.seed(42)
random.seed(42)

# Create folder to save cities
instance_dir = "instances"
os.makedirs(instance_dir, exist_ok=True)

# Load Dataset
df_cities = pd.read_csv("top_cities_coordinates_df.csv")

# Define Instance Sizes
sizes = [10, 15, 20, 30, 50, 75, 100, 200, 300, 400, 500]

# Define Formulations
formulations = {
    "MTZ": run_tsp_gurobi_mtz,
    "DFJ": run_tsp_gurobi_dfj,
    "Flow-Based": run_tsp_gurobi_fb
}

# Set Timeout
timeout = 3600

# Define number of repeats for each instance
repeats = 5

# Run benchmark
rows = []

for formulation_name, solver in formulations.items():
    for size in sizes:

        # Limit DFJ to 20 cities since preprocessing takes too long
        if formulation_name == "DFJ" and size > 20:
            continue

        for rep in range(1, repeats + 1):

            # Create random samples for each repeat
            sample = df_cities.sample(n=size, random_state=rep).reset_index(drop=True)

            # Save those instances
            path = f"{instance_dir}/cities_{formulation_name}_{size}_{rep}.csv"
            sample.to_csv(path, index=False)

            # Run solver
            res = solver(sample, 0, timeout)

            # Save results (Nan if Gurobi found no feasible solution)
            if res["obj"] is None:
                rows.append({
                    "formulation": formulation_name,
                    "n_cities": size,
                    "instance": rep,
                    "runtime": np.nan,
                    "gap in %": np.nan,
                    "obj": np.nan
                })
                continue

            rows.append({
                "formulation": formulation_name,
                "n_cities": size,
                "instance": rep,
                "runtime": res["solve_time"],
                "gap in %": res["gap"] * 100,
                "obj": res["obj"]
            })

df = pd.DataFrame(rows)
df.to_csv("benchmark_results_static.csv", index=False)