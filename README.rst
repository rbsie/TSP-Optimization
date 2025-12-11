Optimization for Travelling Salesman Problem
===========

This Project optimizes the Travelling Salesman Problem (TSP) using different formulations 
(MTZ, DFJ, Flow-Based) and solver engines (PySCIPOpt or Gurobi). The web-based application
allows interactive configuration, visualization, and benchmarking.

Features
--------

* Interactive Streamlit interface
* Choice between random world cities and the cities of the "Eras Tour"
* Multiple TSP formulations for subtour elimination:

  - MTZ (Miller–Tucker–Zemlin)
  - DFJ (Dantzig–Fulkerson–Johnson)
  - Flow-based formulation

* Solver engine selection:

  - PySCIPOpt
  - Gurobi

* Interactive route visualization via Folium
* Benchmark study comparing formulations (Gurobi)

Project Structure
-----------------

The main file is 'tsp_streamlit'.py, where I have implemented all the algorithms and the streamlit application.

The folder 'Datasets' contains the datasets as well as the file I have used to create the data.

In the 'Benchmark' folder, I conducted a small benchmark study to compare the runtime and quality of the different formulations (in Gurobi).

Installation
------------

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/rbsie/TSP-Optimization.git

    # Create and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt


Usage
---------

.. code-block:: python

    streamlit run tsp_streamlit.py

Then open the browser and configure:

1. Select dataset (Random Cities / Eras Tour)  
2. Choose solver formulation  
3. Select solver engine  
4. Set time limit and run the optimization  

A map with the computed tour and a detailed route table will appear.


Notes
-----

* DFJ grows exponentially and will take long for larger instances. I recommend using less than 20 cities for this formulation.
* Gurobi runs faster than PySCIPOpt. To be able to use Gurobi, a licence is needed (https://www.gurobi.com/)

License
-------

This project is intended for private educational use within the COA lecture.
Redistribution or public release is not permitted.