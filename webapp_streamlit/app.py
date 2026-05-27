import streamlit as st
from streamlit.components.v1 import html
import os

# Set directory to this file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Point Reyes Porcini Predictor",
    page_icon="🍄",
    layout="wide"
)

# --- HEADER ---
st.title("Point Reyes Porcini Tracker")
st.markdown("""
    **Current Status:** *Spatial Model Active / Temporal Model Coming Soon* Use this map to identify prime **Bishop Pine** habitat for *Boletus edulis*.
""")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["When to Look", "Where to Look", "Methodology (Data Science)"])

# --- TAB 1: WHEN TO LOOK ---
with tab1:
    st.header("Fruiting Probability (Coming Soon)")
    st.info("This section will host the Temporal Model.")
    
    st.markdown("""
    **Upcoming Logic:**
    1. **Fetch Live Weather:** Pulling daily rain/temp from NOAA for Point Reyes Station.
    2. **Calculate Features:** * *30-Day Cumulative Rain*
        * *Soil Temp Shock (7-day delta)*
    3. **Logistic Regression:** Probability Score (0-100%).
    """)

# --- TAB 2: WHERE TO LOOK ---
with tab2:
    st.header("Potential Habitat Zones")

    # Load the map HTML file
    with open("visualizations/vegetation_trail_map.html", "r", encoding="utf-8") as f:
        map_html = f.read()

    # Display the map using Streamlit's HTML component
    html(map_html, height=600, width=1000)

# --- TAB 3: METHODOLOGY ---
with tab3:
    st.header("How this Model Works")
    st.markdown("""
    ### 1. Spatial Component: Defining the "Where"
    Unlike standard distribution models, this project uses **Ecological Filtering**:
    * **Data Source:** Switched from coarse CALVEG data to the **2018 Marin Fine Scale Vegetation Map** (LIDAR-derived) for canopy-level precision.
    * **Host Filtering:** Strictly isolated *Pinus muricata* (Bishop Pine) and *Pseudotsuga menziesii* (Douglas Fir).
    * **Disturbance Layer:** Integrated **2020 Woodward Fire** perimeter data to exclude burned/dead canopy areas from the prediction set.
    
    ### 2. Temporal Component: Defining the "When"
    *(To be populated after we build your Logistic Regression model)*
    """)

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("Project Info")
    st.write("Created by **Ben Yeager**")
    st.write("Tools: Python, GeoPandas, Folium, Streamlit")
    st.button("Contact Me") # Link to LinkedIn