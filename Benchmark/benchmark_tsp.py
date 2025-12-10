import streamlit as st
import pandas as pd
import numpy as np

def shifted_geom_mean(values, shift=1e-3):
    v = np.array(values) + shift
    return float(np.exp(np.mean(np.log(v))) - shift)

def benchmark_page():
    st.title("Benchmark Study")

    st.write("""
    This benchmark study was precomputed using Gurobi.

    Instance sizes:
    - 10 cities
    - 20 cities
    - 50 cities  
    - 100 cities
    - 200 cities  
    - 500 cities

    For each size and formulation, 5 random instances were tested (with a time limit of 60 seconds).
             
    Since the DFJ formulation causes high run times for larger instances, DFJ is only considered up to 20 cities.
    """)

    df = pd.read_csv("Benchmark/benchmark_results_static.csv")

    st.subheader("Raw Results")
    st.dataframe(df, use_container_width=True)

    # aggregated results
    st.subheader("Aggregated Performance per Formulation and Size")

    agg = (
    df.groupby(["formulation", "n_cities"])
      .agg(
          arith_runtime=("runtime", lambda x: np.nanmean(x)),
          geom_runtime=("runtime", lambda x: shifted_geom_mean(x.dropna())),
          arith_gap=("gap in %", lambda x: np.nanmean(x)),
          geom_gap=("gap in %", lambda x: shifted_geom_mean(x.dropna()))
      )
      .reset_index()
    )

    st.dataframe(agg, use_container_width=True)

    # -----------------------------------------
    # Combined Plot: Runtime (Arithmetic + Geometric)
    # -----------------------------------------

    st.subheader("Runtime Comparison (Arithmetic + Shifted Geometric Mean)")

    tab_a, tab_g = st.tabs(["Arithmetic", "Geometric"])

    with tab_a:
        st.line_chart(agg.pivot(index="n_cities", columns="formulation", values="arith_runtime"))

    with tab_g:
        st.line_chart(agg.pivot(index="n_cities", columns="formulation", values="geom_runtime"))

    # -----------------------------------------
    # Combined Plot: Gap (Arithmetic + Geometric)
    # -----------------------------------------

    st.subheader("Gap Comparison (Arithmetic + Shifted Geometric Mean)")

    tab_a, tab_g = st.tabs(["Arithmetic", "Geometric"])

    with tab_a:
        st.line_chart(agg.pivot(index="n_cities", columns="formulation", values="arith_gap"))

    with tab_g:
        st.line_chart(agg.pivot(index="n_cities", columns="formulation", values="geom_gap"))

    # -----------------------------------------
    # Download option
    # -----------------------------------------

    st.download_button(
        "Download Benchmark Study (CSV)",
        df.to_csv(index=False),
        file_name="Benchmark/benchmark_results_static.csv"
    )

    st.write("""
    For small problems (10 cities), all methods run fast and reach a perfect or near-perfect gap. For DFJ, the runtime increases a lot for more than 20 cities. At 20 cities it still finds solutions, but its gap becomes infinite, meaning that a lower bound coundn't be found. MTZ performs well for small sizes but quickly becomes slow (for > 75 cities) and unstable (for > 100 cities); from 75 cities upward it hits the time limit and the gap grows. The Flow-based model scales much better: it stays fast and accurate up to 100 cities, and even at 200 cities it finds good solutions with a small gap within 60 seconds. For more than 200 cities, both MTZ and Flow  fail to produce consistent results. Overall, Flow-based is the strongest formulation, MTZ is usable only for small to medium sizes, and DFJ is only reliable for very small instances.
    """)