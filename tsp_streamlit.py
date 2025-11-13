"""
TSP Optimization with PySCIPOpt (Streamlit App)
==============================================

A linear optimization model implemented with PySCIPOpt
to solve the Traveling Salesman Problem (TSP).

Features:
- Interactive sidebar for configuration
- Choice of dataset (Eras Tour Cities or Random Cities)
- Choice of solver formulation (MTZ, DFJ, Flow-Based)
- Optimization with PySCIPOpt
- Interactive Folium map visualization
"""

import streamlit as st
import pandas as pd
import folium
from pyscipopt import Model, quicksum
from math import radians, cos, sin, sqrt, atan2
from streamlit_folium import st_folium
from itertools import combinations

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


# ----------------------------------------------------------------------
# Implementation of different formulations: MTZ, DFJ, Flow based
# ----------------------------------------------------------------------

def run_tsp_model_mtz(cities_df, timeout_sec):
    """
    Creates and solves the TSP using the MTZ formulation.
    """
    
    number_cities = len(cities_df)
    city_names = list(cities_df['city'])

    # Define start city (index 0 by default)
    start_city = 0  

    # Calculate distances between all cities
    dist = {(i, j): haversine(cities_df.loc[i, 'lat'], cities_df.loc[i, 'lon'], 
                             cities_df.loc[j, 'lat'], cities_df.loc[j, 'lon']) 
            for i in range(number_cities) 
            for j in range(number_cities) if i != j}

    # Define model
    model = Model("TSP_Tour_MTZ")
    #model.hideOutput()  # hide the solver output

    # --- Define decision variables ---
    # x[i,j] = 1, if there's a path from city i to city j; else 0
    x = {}
    for (i, j) in dist:
        x[i, j] = model.addVar(vtype="B", name=f"x({i},{j})")
    
    # u[i] = position of city i in the tour
    u = {} 
    for i in range(number_cities):
        if i != start_city: # u[start_city] = 1
            u[i] = model.addVar(vtype="I", lb=2, ub=number_cities, name=f"u({i})")

    # --- Objective function ---
    model.setObjective(quicksum(dist[i, j] * x[i, j] for (i, j) in dist), "minimize")

    # --- Constraints ---
    # Each city is left exactly once
    for i in range(number_cities):
        model.addCons(quicksum(x[i, j] for j in range(number_cities) if i != j) == 1)

    # Each city is entered exactly once
    for j in range(number_cities):
        model.addCons(quicksum(x[i, j] for i in range(number_cities) if i != j) == 1)
        
    # Subtour elimination (MTZ)
    for i in range(number_cities):
        for j in range(number_cities):
            if i != j and i != start_city and j != start_city:
                model.addCons(u[i] - u[j] + 1 <= (number_cities - 1)*(1 - x[i, j]))

    # Add timeout
    model.setParam("limits/time", timeout_sec)

    # Start solver
    model.optimize()

    # --- Get results ---
    status = model.getStatus()

    try:
        obj_val = model.getObjVal()
    except Exception:
        obj_val = None

    # Wenn keine Lösung vorliegt → einfache Rückgabe und Meldung
    if obj_val is None or status not in ['optimal', 'timelimit']:
        print('No feasible solution found.')
        return {
            'status': status,
            'obj': None,
            'df': None,
            'route_indices': None,
            'route_names': None,
            'cities_df': cities_df
        }

    if status == 'timelimit':
        status = 'feasible'
        print('Stopped due to time limit.')
    elif status != 'optimal':
        print('No optimal solution found!')

    obj_val = model.getObjVal()
    
    # Reconstruct the route
    edges = [(i, j) for (i, j) in dist if round(model.getVal(x[i, j])) == 1]
    
    route_indices = [start_city]
    current_city = start_city
    
    while len(route_indices) < number_cities:
        found_next = False
        for (i, j) in edges:
            if i == current_city and j not in route_indices:
                route_indices.append(j)
                current_city = j
                found_next = True
                break
    
    route_indices.append(start_city)  # back to the first city
    
    route_names = [city_names[i] for i in route_indices]

    # DataFrame to present the results
    route_df = pd.DataFrame({
        "Step": range(1, len(route_names)),
        "From": route_names[:-1],
        "To": route_names[1:]
    })
    
    return {
        'status': status,            
        'obj': obj_val,
        'df': route_df,
        'route_indices': route_indices,
        'route_names': route_names,
        'cities_df': cities_df
    }


def run_tsp_model_dfj(cities_df, timeout_sec):
    """
    Creates and solves the TSP using the DFJ formulation.
    """

    print('hi')
    
    number_cities = len(cities_df)
    city_names = list(cities_df['city'])

    # Define start city (index 0 by default)
    start_city = 0  

    # Calculate distances between all cities
    dist = {(i, j): haversine(cities_df.loc[i, 'lat'], cities_df.loc[i, 'lon'], 
                             cities_df.loc[j, 'lat'], cities_df.loc[j, 'lon']) 
            for i in range(number_cities) 
            for j in range(number_cities) if i != j}

    # Define model
    model = Model("TSP_Tour_DFJ")
    #model.hideOutput()  # hide the solver output

    # --- Define decision variables ---
    # x[i,j] = 1, if there's a path from city i to city j; else 0
    x = {}
    for (i, j) in dist:
        x[i, j] = model.addVar(vtype="B", name=f"x({i},{j})")

    # --- Objective function ---
    model.setObjective(quicksum(dist[i, j] * x[i, j] for (i, j) in dist), "minimize")

    # --- Constraints ---
    # Each city is left exactly once
    for i in range(number_cities):
        model.addCons(quicksum(x[i, j] for j in range(number_cities) if i != j) == 1)

    # Each city is entered exactly once
    for j in range(number_cities):
        model.addCons(quicksum(x[i, j] for i in range(number_cities) if i != j) == 1)
        
    # Subtour elimination (DFJ)
    # create Subset of size 2,..., n
    for k in range(2, number_cities): 
        for S in combinations(range(number_cities), k):
            model.addCons(quicksum(x[i, j] for i in S for j in S if i != j) <= len(S) - 1)

            print(S)

    # Add timeout
    model.setParam("limits/time", timeout_sec)

    # Start solver
    model.optimize()

    # --- Get results ---
    status = model.getStatus()    

    try:
        obj_val = model.getObjVal()
    except Exception:
        obj_val = None

    # Wenn keine Lösung vorliegt → einfache Rückgabe und Meldung
    if obj_val is None or status not in ['optimal', 'timelimit']:
        print('No feasible solution found.')
        return {
            'status': status,
            'obj': None,
            'df': None,
            'route_indices': None,
            'route_names': None,
            'cities_df': cities_df
        }

    if status == 'timelimit':
        status = 'feasible'
        print('Stopped due to time limit.')
    elif status != 'optimal':
        print('No optimal solution found!')
    
    obj_val = model.getObjVal()
    
    # Reconstruct the route
    edges = [(i, j) for (i, j) in dist if round(model.getVal(x[i, j])) == 1]
    
    route_indices = [start_city]
    current_city = start_city
    
    while len(route_indices) < number_cities:
        found_next = False
        for (i, j) in edges:
            if i == current_city and j not in route_indices:
                route_indices.append(j)
                current_city = j
                found_next = True
                break
    
    route_indices.append(start_city)  # back to the first city
    
    route_names = [city_names[i] for i in route_indices]

    # DataFrame to present the results
    route_df = pd.DataFrame({
        "Step": range(1, len(route_names)),
        "From": route_names[:-1],
        "To": route_names[1:]
    })
    
    return {
        'status': status,            
        'obj': obj_val,
        'df': route_df,
        'route_indices': route_indices,
        'route_names': route_names,
        'cities_df': cities_df
    }


def show_tsp_map(cities_df, route_indices, route_names):
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
    for idx in route_indices:
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
    start_city_data = cities_df.loc[route_indices[0]]
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

st.markdown("""
This app finds the shortest possible route that visits a set of cities
and returns to the starting city. As a city source, you can choose a random selection of the largest cities worldwide. One possible application of the TSP could be to find the best route for a singer or band's concert tour. For this purpose, I have also added the cities from Taylor Swift's "Eras Tour".

Configure your options in the sidebar and run the optimization.
""")

# --- Sidebar Configuration ---
st.sidebar.header('⚙️ Configuration')
cities_to_solve = None
city_data = None
num_cities = 0

# --- 1. Dataset Selection ---
st.sidebar.subheader('1. Select Dataset')

# We use session state to remember which dataset was last used
# to reset the random sample if the user switches datasets.
if 'last_dataset' not in st.session_state:
    st.session_state.last_dataset = None

dataset_option = st.sidebar.radio(
    "Choose your city source:",
    ('Random Cities Worldwide', 'Eras Tour Cities'),
    key='dataset_option'
)

tour_continent = None 

if dataset_option == 'Eras Tour Cities':
    tour_continent = st.sidebar.radio(
        "Choose tour region:",
        ('World', 'Europe', 'United States'),
        key='tour_continent'
    )

# --- Data Loading ---
if dataset_option == 'Eras Tour Cities':
    if tour_continent == 'World': 
        city_data = pd.read_csv('eras_cites_all_df.csv')
    if tour_continent == 'Europe': 
        city_data = pd.read_csv('eras_cites_europe_df.csv')
    if tour_continent == 'United States': 
        city_data = pd.read_csv('eras_cites_US_df.csv')
elif dataset_option == 'Random Cities Worldwide':
    city_data = pd.read_csv('top_cities_coordinates_df.csv')

# --- City Selection Logic ---
max_cities = len(city_data)

# Check if the dataset option has changed. If so, clear the old selection.
if st.session_state.last_dataset != dataset_option:
    st.session_state.last_dataset = dataset_option
    if 'city_selection' in st.session_state:
        del st.session_state.city_selection

# define the possible number of cities for each dataset
if dataset_option == 'Eras Tour Cities':
    num_cities = max_cities
else:
    num_cities = st.sidebar.slider('Number of cities', 
                                min_value=3, 
                                max_value=min(max_cities, 200), 
                                value=8, 
                                step=1)

# --- Handle city selection based on dataset ---
if dataset_option == 'Eras Tour Cities':
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

# --- 2. Solver Selection ---
st.sidebar.subheader('2. Select Solver Method')
solver_method = st.sidebar.radio(
    "Choose the formulation:",
    ('MTZ (Miller-Tucker-Zemlin)', 
     'DFJ (Danzig-Fulkerson-Johnson)', 
     'Flow-Based Formulation'),
    key='solver_method'
)

if solver_method == 'Flow-Based Formulation':
    st.sidebar.warning(f"**{solver_method}** is not yet implemented. Please select MTZ to run.")


# --- 3. Run Controls ---
st.sidebar.subheader('3. Run Optimization')

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

# --- Optimization Run ---
if run_opt:
    # Check if the selected solver is implemented
    if solver_method == 'MTZ (Miller-Tucker-Zemlin)':
        with st.spinner(f'Calculating optimal tour for {num_cities} cities using MTZ... (This may take time)'):
            st.session_state.results = run_tsp_model_mtz(cities_to_solve, timeout_sec)
            if st.session_state.results['status'] != 'optimal':
                st.error(f"Solver problem: {st.session_state.results['status']}")
    elif solver_method == 'DFJ (Danzig-Fulkerson-Johnson)':
        with st.spinner(f'Calculating optimal tour for {num_cities} cities using DFJ... (This may take time)'):
            st.session_state.results = run_tsp_model_dfj(cities_to_solve, timeout_sec)
            if st.session_state.results['status'] != 'optimal':
                st.error(f"Solver problem: {st.session_state.results['status']}")
    else:
        # Handle the placeholder solvers
        st.warning(f"The selected solver **({solver_method})** is not implemented.")
        clear_results()  # Clear any previous results

# ----------------------------------------------------------------------
# Show Results
# ----------------------------------------------------------------------

if st.session_state.results is not None:
    r = st.session_state.results
    if r['obj'] is None:
        st.error(f"No feasible solution found. Solver status: {r['status']}")
    else:
        st.subheader('Model Results')
        col1, col2 = st.columns(2)
        col1.metric("Solver Status", r['status'].capitalize())
        col2.metric("Distance (Tour)", f"{r['obj']:.2f} km")
        st.subheader('🌎 Optimal Route')
        show_tsp_map(r['cities_df'], r['route_indices'], r['route_names'])
        st.subheader('Route Details')
        st.write(f"**Full Tour:** {' → '.join(r['route_names'])}")
        st.dataframe(r['df'])

else:
    st.info('Adjust parameters in the sidebar and click **Run Optimization**.')