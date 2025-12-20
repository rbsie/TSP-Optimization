import streamlit as st
import pandas as pd
import numpy as np

def shifted_geom_mean(values, shift=0.1):
    """Compute the shifted geometric mean of a list of values."""
    v = np.array(values) + shift
    return float(np.exp(np.mean(np.log(v))) - shift)

def benchmark_page():
    st.title("Benchmark Study")

    st.write("""
    This benchmark study was precomputed using Gurobi.

    Instance sizes:
    - 10 cities
    - 15 cities
    - 20 cities
    - 30 cities
    - 50 cities  
    - 100 cities
    - 300 cities
    - 500 cities
             
    For each size and formulation, 5 random instances were tested (same instances across formulations). The time limit was set to 30 minutes (1800 seconds).
             
    Since the DFJ formulation causes high run times for large instances, DFJ is only considered up to 20 cities.
    """)

    # Load benchmark results
    df = pd.read_csv("Benchmark/benchmark_results.csv")

    # Raw results
    st.subheader("Raw Results")
    st.dataframe(df, use_container_width=True)

    # Aggregated results
    st.subheader("Aggregated Performance per Formulation and Size")

    agg = (
    df.groupby(["formulation", "n_cities"]).agg(
        arith_runtime=("runtime", lambda x: np.nanmean(x)), # Arithmetic Mean of runtime
        geom_runtime=("runtime", lambda x: shifted_geom_mean(x.dropna())), # Shifted Geometric Mean of runtime
        arith_gap=("gap in %", lambda x: np.nanmean(x)), # Arithmetic Mean of gap
        geom_gap=("gap in %", lambda x: shifted_geom_mean(x.dropna())) # Shifted Geometric Mean of gap
      ).reset_index()
    )

    st.dataframe(agg, use_container_width=True)

    # Plot: Runtime (Arithmetic + Geometric)
    st.subheader("Runtime Comparison (Arithmetic + Shifted Geometric Mean)")

    tab_a, tab_g = st.tabs(["Arithmetic", "Geometric"])

    with tab_a:
        st.line_chart(agg.pivot(index="n_cities", columns="formulation", values="arith_runtime"))

    with tab_g:
        st.line_chart(agg.pivot(index="n_cities", columns="formulation", values="geom_runtime"))

    # Plot: Gap (Arithmetic + Geometric)
    st.subheader("Gap Comparison (Arithmetic + Shifted Geometric Mean)")

    tab_a, tab_g = st.tabs(["Arithmetic", "Geometric"])

    with tab_a:
        st.line_chart(agg.pivot(index="n_cities", columns="formulation", values="arith_gap"))

    with tab_g:
        st.line_chart(agg.pivot(index="n_cities", columns="formulation", values="geom_gap"))
