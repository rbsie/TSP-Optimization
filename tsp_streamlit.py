"""
TSP Optimization (Streamlit App)

A linear optimization model for solving the Traveling Salesman Problem (TSP).

Features:
- Interactive sidebar for configuration  
- Choice between a random city dataset and the 'Eras Tour' cities
- Multiple TSP formulations (MTZ, DFJ, Flow-Based)  
- Solver engine selection (PySCIPOpt or Gurobi)  
- Interactive Folium map visualization
- Benchmark study for formulation analysis (Gurobi)
"""

import streamlit as st
import pandas as pd
import folium
from math import radians, cos, sin, sqrt, atan2
from streamlit_folium import st_folium
from itertools import combinations
from pyscipopt import Model as SCIPModel, quicksum as scip_quicksum
from gurobipy import Model as GurobiModel, GRB, quicksum as gurobi_quicksum
from Benchmark import benchmark_tsp

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the haversine distance between two cities.
    """
    R = 6371  # Earth radius in km
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def get_city_data(cities_df):
    """
    Returns the number of cities, their names, and pairwise distances.
    """
    number_cities = len(cities_df)
    city_names = list(cities_df['city'])

    # Calculate distances between all cities and save in dictionary
    dist = {(i, j): haversine(cities_df.loc[i, 'lat'], cities_df.loc[i, 'lon'], 
                             cities_df.loc[j, 'lat'], cities_df.loc[j, 'lon']) 
            for i in range(number_cities) 
            for j in range(number_cities) if i != j}

    return number_cities, city_names, dist

# ----------------------------------------------------------------------
# Implementation of different formulations: MTZ, DFJ, Flow based
# ----------------------------------------------------------------------
# I. PySCIPOpt Implementation
# ----------------------------------------------------------------------

def run_tsp_scip_mtz(cities_df, start_city, timeout_sec):
    """
    Creates and solves the TSP using the MTZ formulation (PySCIPOpt).
    """
    # City data
    number_cities, city_names, dist = get_city_data(cities_df)
    
    # Define model 
    model = SCIPModel("TSP_Tour_MTZ_PySCIPOpt")

    # Define decision variables
    # x[i,j] = 1, if there's a path from city i to city j; else 0
    x = {}
    for (i, j) in dist:
        x[i, j] = model.addVar(vtype="B", name=f"x({i},{j})")

    # Constraints
    # Each city is left exactly once
    for i in range(number_cities):
        model.addCons(scip_quicksum(x[i, j] for j in range(number_cities) if i != j) == 1)

    # Each city is entered exactly once
    for j in range(number_cities):
        model.addCons(scip_quicksum(x[i, j] for i in range(number_cities) if i != j) == 1)

    # Objective function
    model.setObjective(scip_quicksum(dist[i, j] * x[i, j] for (i, j) in dist), "minimize")
    
    # Subtour elimination (MTZ)
    # u[i] = position of city i in the tour (for all cities except start city)
    u = {} 
    for i in range(number_cities):
        if i != start_city:
            u[i] = model.addVar(vtype="C", lb=2, ub=number_cities, name=f"u({i})")
        
    for i in range(number_cities):
        for j in range(number_cities):
            if i != j and i != start_city and j != start_city:
                model.addCons(u[i] - u[j] + number_cities * x[i, j] <= number_cities - 1)

    # Timeout
    model.setParam("limits/time", timeout_sec)

    # Start solver
    model.optimize()

    # Get results
    status = model.getStatus() # either optimal or timeout
    obj_val = model.getObjVal() # length of the route in km
    
    # Reconstruct the route
    edges = [(i, j) for (i, j) in dist if model.getVal(x[i, j]) == 1]

    route = [start_city]
    current = start_city
    
    while len(route) < number_cities:
        for i, j in edges:
            if i == current and j not in route:
                route.append(j)
                current = j
                break
        
    route.append(start_city)  # back to the first city
    route_names = [city_names[i] for i in route]

    # DataFrame to show the results
    route_df = pd.DataFrame({
        "Step": range(1, len(route_names)),
        "From": route_names[:-1],
        "To": route_names[1:]
    })

    # Information about Solving
    solve_time = model.getSolvingTime()
    nodes = model.getNNodes()
    primal = model.getPrimalbound()
    dual = model.getDualbound()
    gap = model.getGap()

    return {
        'status': status,
        'obj': obj_val,
        'df': route_df,
        'route': route,
        'route_names': route_names,
        'cities_df': cities_df,
        'solve_time': solve_time,
        'nodes': nodes,
        'primal': primal,
        'dual': dual,
        'gap': gap
    }

def run_tsp_scip_dfj(cities_df, start_city, timeout_sec):
    """
    Creates and solves the TSP using the DFJ formulation (PySCIPOpt).
    """
    # City data 
    number_cities, city_names, dist = get_city_data(cities_df)

    # Define model
    model = SCIPModel("TSP_Tour_DFJ_PySCIPOpt")

    # Define decision variables
    # x[i,j] = 1, if there's a path from city i to city j; else 0
    x = {}
    for (i, j) in dist:
        x[i, j] = model.addVar(vtype="B", name=f"x({i},{j})")

    # Constraints
    # Each city is left exactly once
    for i in range(number_cities):
        model.addCons(scip_quicksum(x[i, j] for j in range(number_cities) if i != j) == 1)

    # Each city is entered exactly once
    for j in range(number_cities):
        model.addCons(scip_quicksum(x[i, j] for i in range(number_cities) if i != j) == 1)

    # Objective function
    model.setObjective(scip_quicksum(dist[i, j] * x[i, j] for (i, j) in dist), "minimize")
    
    # Subtour elimination (DFJ)
    # create Subset of size 2,..., n-1
    for k in range(2, number_cities): 
        for S in combinations(range(number_cities), k):
            model.addCons(scip_quicksum(x[i, j] for i in S for j in S if i != j) <= len(S) - 1)
            print(S)

    # Timeout
    model.setParam("limits/time", timeout_sec)

    # Start solver
    model.optimize()

    # Get results
    status = model.getStatus()    
    obj_val = model.getObjVal()
    
    # Reconstruct the route
    edges = [(i, j) for (i, j) in dist if model.getVal(x[i, j]) == 1]

    route = [start_city]
    current = start_city
    
    while len(route) < number_cities:
        for i, j in edges:
            if i == current and j not in route:
                route.append(j)
                current = j
                break
        
    route.append(start_city)  # back to the first city
    route_names = [city_names[i] for i in route]
    
    # DataFrame to present the results
    route_df = pd.DataFrame({
        "Step": range(1, len(route_names)),
        "From": route_names[:-1],
        "To": route_names[1:]
    })
    
    # Information about Solving
    solve_time = model.getSolvingTime()
    nodes = model.getNNodes()
    primal = model.getPrimalbound()
    dual = model.getDualbound()
    gap = model.getGap()

    return {
        'status': status,
        'obj': obj_val,
        'df': route_df,
        'route': route,
        'route_names': route_names,
        'cities_df': cities_df,
        'solve_time': solve_time,
        'nodes': nodes,
        'primal': primal,
        'dual': dual,
        'gap': gap
    }

def run_tsp_scip_fb(cities_df, start_city, timeout_sec):
    """
    Creates and solves the TSP using the Flow-Based formulation (PySCIPOpt).
    """
    # City data 
    number_cities, city_names, dist = get_city_data(cities_df) 

    # Define model
    model = SCIPModel("TSP_Tour_FB_PySCIPOpt")

    # Define decision variables
    # x[i,j] = 1, if there's a path from city i to city j; else 0
    x = {}
    for (i, j) in dist:
        x[i, j] = model.addVar(vtype="B", name=f"x({i},{j})")

    # Constraints
    # Each city is left exactly once
    for i in range(number_cities):
        model.addCons(scip_quicksum(x[i, j] for j in range(number_cities) if i != j) == 1)

    # Each city is entered exactly once
    for j in range(number_cities):
        model.addCons(scip_quicksum(x[i, j] for i in range(number_cities) if i != j) == 1)

    # Objective function
    model.setObjective(scip_quicksum(dist[i, j] * x[i, j] for (i, j) in dist), "minimize")
        
    # Subtour elimination (Flow-Based)
    # Flow variable
    f = {}
    for (i, j) in dist:
        f[i, j] = model.addVar(vtype="C", lb=0, ub=number_cities - 1)

    # Net outflow = n-1 for the start city
    model.addCons(
        scip_quicksum(f[start_city, j] for (k, j) in dist if k == start_city)
        == number_cities - 1
    )

    # Flow balance for all cities except the start city
    for i in range(number_cities):
        if i != start_city:
            model.addCons(
                # inflow to i
                scip_quicksum(f[j, i] for (j, k) in dist if k == i)
                -
                # outflow from i
                scip_quicksum(f[i, j] for (k, j) in dist if k == i)
                == 1
            )

    # Flow only on used edges
    for (i, j) in dist:
        model.addCons(f[i, j] <= (number_cities - 1) * x[i, j])

    # Timeout
    model.setParam("limits/time", timeout_sec)

    # Start solver
    model.optimize()

    # Get results
    status = model.getStatus()    
    obj_val = model.getObjVal()
    
    # Reconstruct the route
    edges = [(i, j) for (i, j) in dist if model.getVal(x[i, j]) == 1]

    route = [start_city]
    current = start_city
    
    while len(route) < number_cities:
        for i, j in edges:
            if i == current and j not in route:
                route.append(j)
                current = j
                break
        
    route.append(start_city)  # back to the first city
    route_names = [city_names[i] for i in route]

    # DataFrame to present the results
    route_df = pd.DataFrame({
        "Step": range(1, len(route_names)),
        "From": route_names[:-1],
        "To": route_names[1:]
    })
    
    # Information about Solving
    solve_time = model.getSolvingTime()
    nodes = model.getNNodes()
    primal = model.getPrimalbound()
    dual = model.getDualbound()
    gap = model.getGap()

    return {
        'status': status,
        'obj': obj_val,
        'df': route_df,
        'route': route,
        'route_names': route_names,
        'cities_df': cities_df,
        'solve_time': solve_time,
        'nodes': nodes,
        'primal': primal,
        'dual': dual,
        'gap': gap
    }

# ----------------------------------------------------------------------
# II. Gurobi Implementation
# ----------------------------------------------------------------------

def run_tsp_gurobi_mtz(cities_df, start_city, timeout_sec):
    """
    Creates and solves the TSP using the MTZ formulation (Gurobi).
    """
    # City data
    number_cities, city_names, dist = get_city_data(cities_df)

    # Model
    model = GurobiModel("TSP_Tour_MTZ_Gurobi")

    # Set timeout
    model.Params.TimeLimit = timeout_sec

    # Decision variables
    x = {}
    for (i, j) in dist:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x({i},{j})")

    # Objective Function
    model.setObjective(gurobi_quicksum(dist[i,j] * x[i,j] for (i,j) in dist), GRB.MINIMIZE)

    # Constraints
    # Each city is left exactly once
    for i in range(number_cities):
        model.addConstr(gurobi_quicksum(x[i,j] for j in range(number_cities) if j != i) == 1)

    # Each city is entered exactly once
    for j in range(number_cities):
        model.addConstr(gurobi_quicksum(x[i,j] for i in range(number_cities) if i != j) == 1)

    # MTZ subtour elimination
    u = {}
    for i in range(number_cities):
        if i != start_city:
            u[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=2, ub=number_cities, name=f"u({i})")

    for i in range(number_cities):
        for j in range(number_cities):
            if i != j and i != start_city and j != start_city:
                model.addConstr(u[i] - u[j] + number_cities * x[i, j] <= number_cities - 1)

    # Start solver
    model.optimize()

    # Get results
    if model.SolCount == 0:
        return {'status': 'infeasible', 'obj': None} # no feasible solution found
    
    status = model.Status # either GRB.OPTIMAL or GRB.TIME_LIMIT
    obj_val = model.ObjVal # length of the route in km

    # Reconstruct the route
    edges = [(i, j) for (i, j) in dist if x[i, j].X == 1]

    route = [start_city]
    current = start_city
    
    while len(route) < number_cities:
        for i, j in edges:
            if i == current and j not in route:
                route.append(j)
                current = j
                break
        
    route.append(start_city)  # back to the first city
    route_names = [city_names[i] for i in route]

    # DataFrame to present the results
    route_df = pd.DataFrame({
        "Step": range(1, len(route_names)),
        "From": route_names[:-1],
        "To": route_names[1:]
    })

    # Information about Solving
    solve_time = model.Runtime
    nodes = model.NodeCount
    primal = model.ObjVal
    dual = model.ObjBound
    gap = model.MIPGap

    return {
        'status': status,
        'obj': obj_val,
        'df': route_df,
        'route': route,
        'route_names': route_names,
        'cities_df': cities_df,
        'solve_time': solve_time,
        'nodes': nodes,
        'primal': primal,
        'dual': dual,
        'gap': gap
    }

def run_tsp_gurobi_dfj(cities_df, start_city, timeout_sec):
    """
    Creates and solves the TSP using the DFJ formulation (Gurobi).
    """
    # City Data
    number_cities, city_names, dist = get_city_data(cities_df)

    # Model
    model = GurobiModel("TSP_Tour_DFJ_Gurobi")

    # Set timeout
    model.Params.TimeLimit = timeout_sec

    # Decision Variables
    x = {}
    for (i, j) in dist:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x({i},{j})")

    # Objective Function
    model.setObjective(gurobi_quicksum(dist[i,j] * x[i,j] for (i,j) in dist), GRB.MINIMIZE)

    # Constraints
    # Each city is left exactly once
    for i in range(number_cities):
        model.addConstr(gurobi_quicksum(x[i,j] for j in range(number_cities) if j != i) == 1)

    # Each city is entered exactly once
    for j in range(number_cities):
        model.addConstr(gurobi_quicksum(x[i,j] for i in range(number_cities) if i != j) == 1)

    # DFJ subtour elimination
    for k in range(2, number_cities):
        for S in combinations(range(number_cities), k):
            model.addConstr(
                gurobi_quicksum(x[i,j] for i in S for j in S if i != j) <= len(S) - 1
            )

    # Start solver
    model.optimize()

    # Get results
    if model.SolCount == 0:
        return {'status': 'infeasible', 'obj': None} # no feasible solution found
    
    status = model.Status # either GRB.OPTIMAL or GRB.TIME_LIMIT
    obj_val = model.ObjVal # length of the route in km

    # Reconstruct the route
    edges = [(i, j) for (i, j) in dist if x[i, j].X == 1]

    route = [start_city]
    current = start_city
    
    while len(route) < number_cities:
        for i, j in edges:
            if i == current and j not in route:
                route.append(j)
                current = j
                break
        
    route.append(start_city)  # back to the first city
    route_names = [city_names[i] for i in route]

    route_df = pd.DataFrame({
        "Step": range(1, len(route_names)),
        "From": route_names[:-1],
        "To": route_names[1:]
    })

    # Information about Solving
    solve_time = model.Runtime
    nodes = model.NodeCount
    primal = model.ObjVal
    dual = model.ObjBound
    gap = model.MIPGap

    return {
        'status': status,
        'obj': obj_val,
        'df': route_df,
        'route': route,
        'route_names': route_names,
        'cities_df': cities_df,
        'solve_time': solve_time,
        'nodes': nodes,
        'primal': primal,
        'dual': dual,
        'gap': gap
    }

def run_tsp_gurobi_fb(cities_df, start_city, timeout_sec):
    """
    Creates and solves the TSP using the Flow-Based formulation (Gurobi).
    """
    # City Data
    number_cities, city_names, dist = get_city_data(cities_df)

    # Model
    model = GurobiModel("TSP_Tour_FB_Gurobi")

    # Set timeout
    model.Params.TimeLimit = timeout_sec

    # Variables
    x = {}
    for (i, j) in dist:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x({i},{j})")

    # Objective
    model.setObjective(gurobi_quicksum(dist[i,j] * x[i,j] for (i,j) in dist), GRB.MINIMIZE)

    # Constraints
    # Each city is left exactly once
    for i in range(number_cities):
        model.addConstr(gurobi_quicksum(x[i,j] for j in range(number_cities) if j != i) == 1)

    # Each city is entered exactly once
    for j in range(number_cities):
        model.addConstr(gurobi_quicksum(x[i,j] for i in range(number_cities) if i != j) == 1)

    # Flow Variable
    f = {(i,j): model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=number_cities-1, name=f"f({i},{j})")
         for (i,j) in dist}
    
    # Net outflow = n-1 for the start city
    model.addConstr(
        gurobi_quicksum(f[start_city,j] for (k,j) in dist if k == start_city)
        == number_cities - 1
    )
    
    # Flow balance for all cities except the start city
    for i in range(number_cities):
        if i != start_city:
            model.addConstr(
                # inflow to i
                gurobi_quicksum(f[j, i] for (j, k) in dist if k == i)
                -
                # outflow from i
                gurobi_quicksum(f[i, j] for (k, j) in dist if k == i)
                == 1
            )

    # Flow only on selected edges
    for (i,j) in dist:
        model.addConstr(f[i,j] <= (number_cities - 1) * x[i,j])

    # Start solver
    model.optimize()

    # Get results
    if model.SolCount == 0:
        return {'status': 'infeasible', 'obj': None} # no feasible solution found
    
    status = model.Status # either GRB.OPTIMAL or GRB.TIME_LIMIT
    obj_val = model.ObjVal # length of the route in km

    # Reconstruct the route
    edges = [(i, j) for (i, j) in dist if x[i, j].X == 1]

    route = [start_city]
    current = start_city
    
    while len(route) < number_cities:
        for i, j in edges:
            if i == current and j not in route:
                route.append(j)
                current = j
                break
        
    route.append(start_city)  # back to the first city
    route_names = [city_names[i] for i in route]

    route_df = pd.DataFrame({
        "Step": range(1, len(route_names)),
        "From": route_names[:-1],
        "To": route_names[1:]
    })

    # Information about Solving
    solve_time = model.Runtime
    nodes = model.NodeCount
    primal = model.ObjVal
    dual = model.ObjBound
    gap = model.MIPGap

    return {
        'status': status,
        'obj': obj_val,
        'df': route_df,
        'route': route,
        'route_names': route_names,
        'cities_df': cities_df,
        'solve_time': solve_time,
        'nodes': nodes,
        'primal': primal,
        'dual': dual,
        'gap': gap
    }

def show_tsp_map(cities_df, route, route_names):
    """
    Visualizes the TSP route on a Folium map.
    """
    
    # Calculate map center
    avg_lat = cities_df['lat'].mean()
    avg_lon = cities_df['lon'].mean()
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=2)

    # Markers for all cities
    for _, row in cities_df.iterrows():
        folium.Marker(
            [row['lat'], row['lon']],
            popup=f"{row['city']}",
            icon=folium.Icon(color='blue', icon='circle', prefix='fa')
        ).add_to(m)

    # Collect route coordinates
    route_coords = []
    for idx in route:
        city = cities_df.loc[idx]
        route_coords.append([city['lat'], city['lon']])
    
    # Polyline for the route
    folium.PolyLine(
        route_coords,
        color='red',
        weight=3,
        opacity=0.8,
        popup=f"Optimal Route: {' → '.join(route_names)}"
    ).add_to(m)
    
    # Highlight start city
    start_city_data = cities_df.loc[route[0]]
    folium.Marker(
        [start_city_data['lat'], start_city_data['lon']],
        popup=f"Start/End: {start_city_data['city']}",
        icon=folium.Icon(color='green', icon='flag', prefix='fa')
    ).add_to(m)

    st_folium(m, width=800, height=500)


# ----------------------------------------------------------------------
# Streamlit App Layout
# ----------------------------------------------------------------------

st.set_page_config(page_title='TSP Optimization', layout='wide')
st.title('Traveling Salesman Problem (TSP) Optimization')

# Sidebar Configuration 
st.sidebar.header('⚙️ Configuration')
cities_to_solve = None
city_data = None
num_cities = 0

page = st.sidebar.radio(
    "Navigation",
    ["TSP Solver", "Benchmark Study"]
)

if page == "TSP Solver":

    st.markdown("""
    This app finds the shortest possible route that visits a set of cities
    and returns to the starting city. As a city source, you can choose a random selection of the largest cities worldwide. One possible application of the TSP could be to find the best route for a singer or band's concert tour. For this purpose, I have also added the cities from Taylor Swift's "Eras Tour".

    Configure your options in the sidebar and run the optimization.
    """)

    # Dataset Selection
    st.sidebar.subheader('1. Select Dataset')

    # We use session state to remember which dataset was last used
    # to reset the random sample if the user switches datasets
    if 'last_dataset' not in st.session_state:
        st.session_state.last_dataset = None

    dataset_option = st.sidebar.radio(
        "Choose your city source:",
        ('Random Cities Worldwide', '"Eras Tour" Cities'),
        key='dataset_option'
    )

    tour_continent = None 

    if dataset_option == '"Eras Tour" Cities':
        tour_continent = st.sidebar.radio(
            "Choose tour region:",
            ('World', 'Europe', 'United States'),
            key='tour_continent'
        )

    # Data Loading
    if dataset_option == '"Eras Tour" Cities':
        if tour_continent == 'World': 
            city_data = pd.read_csv('Datasets/eras_cites_all_df.csv')        # 51
        if tour_continent == 'Europe': 
            city_data = pd.read_csv('Datasets/eras_cites_europe_df.csv')     # 18
        if tour_continent == 'United States': 
            city_data = pd.read_csv('Datasets/eras_cites_US_df.csv')         # 23
    elif dataset_option == 'Random Cities Worldwide':
        city_data = pd.read_csv('Datasets/top_cities_coordinates_df.csv')    # 727

    # City Selection Logic
    max_cities = len(city_data)

    # Check if the dataset option has changed. If so, clear the old selection.
    if st.session_state.last_dataset != dataset_option:
        st.session_state.last_dataset = dataset_option
        if 'city_selection' in st.session_state:
            del st.session_state.city_selection

    # Define the possible number of cities for each dataset
    if dataset_option == '"Eras Tour" Cities':
        num_cities = max_cities
    else:
        num_cities = st.sidebar.slider('Number of cities', 
                                    min_value=3, 
                                    max_value=min(max_cities, 500), 
                                    value=8, 
                                    step=1)

    # Handle city selection based on dataset
    if dataset_option == '"Eras Tour" Cities':
        cities_to_solve = city_data
        # Clear any random selection from session state
        if 'city_selection' in st.session_state:
            del st.session_state.city_selection

    elif dataset_option == 'Random Cities Worldwide':
        # Random sampling behavior
        
        # Add a button to manually resample
        if st.sidebar.button('🎲 Resample Worldwide Cities'):
            if 'city_selection' in st.session_state:
                del st.session_state.city_selection
        
        # Check if a selection exists in state AND if the number of cities matches
        if 'city_selection' not in st.session_state or len(st.session_state.city_selection) != num_cities:
            # If not, create a new random sample and store it
            st.session_state.city_selection = city_data.sample(n=num_cities).reset_index(drop=True)
            
        # Use the stored selection
        cities_to_solve = st.session_state.city_selection

    # Display the selected cities
    selected_cities = cities_to_solve[['city']].copy()
    selected_cities.index = selected_cities.index + 1

    st.sidebar.subheader('Selected Cities:')
    st.sidebar.dataframe(selected_cities, height=200)

    # Choose Start City
    st.sidebar.subheader("Start City")
    start_city = st.sidebar.selectbox(
        "Choose start city:",
        options=list(range(len(cities_to_solve))),
        format_func=lambda i: cities_to_solve.loc[i, "city"],
        index=0
    )

    # Solver Selection 
    st.sidebar.subheader('2. Select Solver Method')
    solver_method = st.sidebar.radio(
        "Choose the formulation:",
        ('MTZ (Miller-Tucker-Zemlin)', 
        'DFJ (Danzig-Fulkerson-Johnson)', 
        'Flow-Based Formulation'),
        key='solver_method'
    )

    # Engine Selection
    st.sidebar.subheader('3. Select Solver Engine')
    solver_engine = st.sidebar.radio(
        "Choose the Solver Engine:",
        ("PySCIPOpt", "Gurobi"), help='Gurobi is a faster Solver Engine.'
    )

    # Run Controls
    st.sidebar.subheader('4. Run Optimization')

    # Add Timout
    timeout_sec = st.sidebar.slider('⏱️ Time limit (seconds)', 5, 100, 10, step=1, help='For a large number of cities the solver may take a long time. It may be useful to set a time limit.')


    def clear_results():
        """
        Resets the session state.
        """
        st.session_state.results = None

    st.sidebar.button('🔄 Reset', on_click=clear_results)
    run_opt = st.sidebar.button('▶️ Run Optimization')

    if 'results' not in st.session_state:
        st.session_state.results = None

    # Optimization Run
    if run_opt:
        if solver_engine == "PySCIPOpt":
            match solver_method:
                case 'MTZ (Miller-Tucker-Zemlin)':
                    with st.spinner(f'Calculating optimal tour for {num_cities} cities using MTZ...'):
                        st.session_state.results = run_tsp_scip_mtz(cities_to_solve, start_city, timeout_sec)

                case 'DFJ (Danzig-Fulkerson-Johnson)':
                    with st.spinner(f'Calculating optimal tour for {num_cities} cities using DFJ...'):
                        st.session_state.results = run_tsp_scip_dfj(cities_to_solve, start_city, timeout_sec)

                case 'Flow-Based Formulation':
                    with st.spinner(f'Calculating optimal tour for {num_cities} cities using the Flow-Based Formulation...'):
                        st.session_state.results = run_tsp_scip_fb(cities_to_solve, start_city, timeout_sec)
        else:  # Gurobi
            match solver_method:
                case 'MTZ (Miller-Tucker-Zemlin)':
                    with st.spinner(f'Calculating optimal tour for {num_cities} cities using MTZ...'):
                        st.session_state.results = run_tsp_gurobi_mtz(cities_to_solve, start_city, timeout_sec)

                case 'DFJ (Danzig-Fulkerson-Johnson)':
                    with st.spinner(f'Calculating optimal tour for {num_cities} cities using DFJ...'):
                        st.session_state.results = run_tsp_gurobi_dfj(cities_to_solve, start_city, timeout_sec)

                case 'Flow-Based Formulation':
                    with st.spinner(f'Calculating optimal tour for {num_cities} cities using the Flow-Based Formulation...'):
                        st.session_state.results = run_tsp_gurobi_fb(cities_to_solve, start_city, timeout_sec)

# ----------------------------------------------------------------------
# Show Results
# ----------------------------------------------------------------------

    GUROBI_STATUS = {
        2: "Optimal",
        3: "INFEASIBLE",
        9: "Time Limit reached"
    }

    if st.session_state.results is not None:
        r = st.session_state.results
        if r['obj'] is None and r['status'] != 'feasible':
            st.error(f"Timeout was reached. No feasible solution found yet.")
        else:
            st.subheader('Model Results')
            col1, col2 = st.columns(2)
            if isinstance(r['status'], int):   # Gurobi
                status_display = GUROBI_STATUS.get(r['status'], f"Code {r['status']}")
            else:                              # PySCIPOpt
                status_display = r['status'].capitalize()

            col1.metric("Solver Status", status_display)

            col2.metric("Distance (Tour)", f"{r['obj']:.2f} km")
            # Information about Solving
            st.text(
                f"Solving Time: {r['solve_time']:.2f} s\n"
                f"Nodes: {r['nodes']}\n"
                f"Primal Bound: {r['primal']:.2f}\n"
                f"Dual Bound: {r['dual']:.2f}\n"
                f"Gap: {r['gap']*100:.2f} %"
            )
            st.subheader('🌎 Optimal Route')
            show_tsp_map(r['cities_df'], r['route'], r['route_names'])
            st.subheader('Route Details')
            st.write(f"**Full Tour:** {' → '.join(r['route_names'])}")
            st.dataframe(r['df'])

elif page == "Benchmark Study":
    benchmark_tsp.benchmark_page()