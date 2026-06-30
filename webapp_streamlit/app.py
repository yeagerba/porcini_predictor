import streamlit as st
from streamlit.components.v1 import html
import os
import pandas as pd
import geopandas as gpd
import altair as alt
import datetime as dt
import json
import time
import folium
from streamlit_folium import st_folium


# Set directory to this file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Point Reyes Porcini Predictor",
    page_icon="🍄",
    layout="wide"
)

# --- HEADER ---
st.title("Point Reyes Porcini Forecast")
# st.markdown("""
#     **Current Status:** *Spatial Model Active / Temporal Model Coming Soon* Use this map to identify prime **Bishop Pine** habitat for *Boletus edulis*.
# """)

# --- TABS ---
# tab1, tab2, tab3 = st.tabs(["Forecast", "Where to Look", "Methodology (Data Science)"])
tab1, tab3, tab4 = st.tabs(["Current Forecast", "Historical Data & Where to Find Porcini", "Contact"])

# --- TAB 1: FORECAST ---
with tab1:
    # st.header("OVERVIEW")

    # Load the predictions
    timestamp = int(time.time()) # Add timestamp to the URL to avoid caching
    gcs_url = "https://storage.googleapis.com/point-reyes-mushroom-data/predictions.csv?t={timestamp}"
    predictions_df = pd.read_csv(gcs_url)
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    # # Filter to only show today onwards
    # today = pd.Timestamp(dt.date.today())
    # # filtered_df = predictions_df[predictions_df['date'] >= today]
    # # TESTING
    # filtered_df = predictions_df[(predictions_df['date'] >= pd.Timestamp(dt.date(2025, 12, 24))) & (predictions_df['date'] <= pd.Timestamp(dt.date(2025, 12, 24) + dt.timedelta(days=14)))]

    # Create info text explaining the forecast
    likelihood_dict = {
        20: 'LOW',
        50: 'ELEVATED',
        70: 'HIGH',
        100: 'VERY HIGH'
    }
    likelihood_text = "Likelihood of porcini fruiting is currently "
    for key, value in likelihood_dict.items():
        if predictions_df['porcini_index'].iloc[0] <= key:
            likelihood_text += value
            current_likelihood = value
            break
    likelihood_text += ".\n\n"
    likelihood_text += "Likelihood of porcini fruiting within the next two weeks is "
    for key, value in likelihood_dict.items():
        if predictions_df['porcini_index'].max() <= key:
            likelihood_text += value
            forecast_likelihood = value
            break
    likelihood_text += ".\n\n"

    if current_likelihood == 'LOW' and forecast_likelihood == 'LOW' and predictions_df['date'].iloc[0].month in [4, 5, 6, 7, 8, 9]:
        infotext = """
        **No porcini expected in forecast period. Porcini season in Point Reyes runs from Oct-Feb. Check back later!**
        
        In the meantime, explore historical data on the next tab to analyze the timing and locations of porcini sightings.
        """
   
    
    st.info(infotext, icon="😴")
    st.text(likelihood_text)


    # st.header("FORECAST")


    # ================================================
    # FORECAST CHART
    # ================================================

    # Create two columns to prevent the chart from overflowing the screen
    col1, col2 = st.columns([4, 1])
    with col1:
        # To label dates only on the top (porcini) and bottom (rain) charts,
        # we set the x-axis for those charts, and omit from temp.

        # Shared X spec for charts with labels at top
        date_x_axis_top = alt.X(
            'date:T',
            axis=alt.Axis(format='%a %d', title=None, grid=False, labelAngle=-45, tickCount=14, orient='top')
        )
        # Shared X spec for charts with labels at bottom
        date_x_axis_bottom = alt.X(
            'date:T',
            axis=alt.Axis(format='%a %d', title=None, grid=False, labelAngle=-45, tickCount=14, orient='bottom')
        )
        # Shared X spec for charts without labels
        date_x_no_labels = alt.X(
            'date:T',
            axis=alt.Axis(labels=False, ticks=False, title=None, grid=False)
        )

        # Create a base layer for the porcini chart (with x labels at top)
        base_top = alt.Chart(predictions_df).encode(
            x=date_x_axis_top
        )
        # Create a base layer for the mid chart (no x labels)
        base_mid = alt.Chart(predictions_df).encode(
            x=date_x_no_labels
        )
        # Create a base layer for the bottom/rain chart (x labels at bottom)
        base_bottom = alt.Chart(predictions_df).encode(
            x=date_x_axis_bottom
        )

        # Define bands for background annotation (for colored zones only, no text labels on bands)
        bands = [
            {"y0": 0, "y1": 10, "color": "#CFD6EA"},
            {"y0": 10, "y1": 40, "color": "#EDF7ED"},
            {"y0": 40, "y1": 70, "color": "#D3EBCD"},
            {"y0": 70, "y1": 100, "color": "#AED5A2"},
        ]
        band_chart = alt.Chart(pd.DataFrame(bands)).mark_rect().encode(
            y='y0:Q',
            y2='y1:Q',
            color=alt.Color('color:N', scale=None, legend=None),
        ).properties(
            width='container',
            height=150
        )

        # Add thin dark gray horizontal lines at each y tick
        yticks = [0, 10, 40, 70, 100]
        ytick_labels = {
            0: "LOW",
            10: "ELEVATED",
            40: "HIGH",
            70: "VERY HIGH",
            100: ""
        }
        lines_df = pd.DataFrame({'y': yticks})
        lines_chart = alt.Chart(lines_df).mark_rule(strokeWidth=1, color='#444', opacity=0.65).encode(
            y='y:Q'
        ).properties(
            width='container',
            height=150
        )

        porcini_line = base_top.mark_line(
            color='#5C3A21', strokeWidth=3, interpolate='monotone'
        ).encode(
            y=alt.Y(
                'porcini_index:Q',
                title='Porcini Abundance',
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(
                    grid=False,
                    values=yticks,
                    tickCount=len(yticks),
                    ticks=True,
                    labels=True,
                    labelOverlap=False,
                    labelExpr=f"""
                        {{
                            0: '{ytick_labels[0]}',
                            10: '{ytick_labels[10]}',
                            40: '{ytick_labels[40]}',
                            70: '{ytick_labels[70]}',
                            100: '{ytick_labels[100]}'
                        }}[datum.value] || ''
                    """,
                    labelBaseline="bottom",   # Move the labels up above the ticks
                    labelPadding=0            # Reduce padding (default is 4), move closer to axis (and up)
                )
            )
        ).properties(height=150)

        porcini_points = base_top.mark_point(
            color='#5C3A21', filled=True, size=60
        ).encode(
            y=alt.Y('porcini_index:Q')
        ).properties(height=150)

        porcini = band_chart + lines_chart + porcini_line + porcini_points

        # Create a DataFrame for grid lines at 0, 5, 10, 15, 20, 25
        temp_grid_y = [0, 5, 10, 15, 20, 25]
        temp_grid_df = pd.DataFrame({'y': temp_grid_y})
        temp_grid_chart = alt.Chart(temp_grid_df).mark_rule(
            strokeWidth=1,
            color='#333',
            opacity=0.45
        ).encode(
            y='y:Q'
        ).properties(
            height=150
        )

        temp = (
            (temp_grid_chart
            + base_mid.mark_line(
                color='#FF4B4B', strokeWidth=1.5, interpolate='monotone'
            ).encode(
                y=alt.Y(
                    'tmax_c_true:Q',
                    title='Temp (°C)',
                    scale=alt.Scale(domain=[0, 25]),
                    axis=alt.Axis(grid=False)
                )
            ).properties(height=150)
            + base_mid.mark_point(
                color='#FF4B4B', filled=True, size=30
            ).encode(
                y=alt.Y('tmax_c_true:Q', scale=alt.Scale(domain=[0, 25]))
            ).properties(height=150)
            + base_mid.mark_line(
                color='#0077B6', strokeWidth=1.5, interpolate='monotone'
            ).encode(
                y=alt.Y(
                    'tmin_c_true:Q',
                    title='Temp (°C)',
                    scale=alt.Scale(domain=[0, 25]),
                    axis=alt.Axis(grid=False)
                )
            ).properties(height=150)
            + base_mid.mark_point(
                color='#0077B6', filled=True, size=30
            ).encode(
                y=alt.Y('tmin_c_true:Q', scale=alt.Scale(domain=[0, 25]))
            ).properties(height=150)
            )
        )

        # Set rain y-axis domain: [0, max(10, prcp_mm_true.max())]
        rain_y_max = max(10, predictions_df['prcp_mm_true'].max())
        rain = base_bottom.mark_bar(
            color='#3A86FF', opacity=0.8
        ).encode(
            y=alt.Y(
                'prcp_mm_true:Q',
                title='Rain (mm)',
                scale=alt.Scale(domain=[0, rain_y_max]),
                axis=alt.Axis(grid=False)
            )
        ).properties(height=150)

        # If no rain is forecasted at all, overlay gray text
        if predictions_df['prcp_mm_true'].max() == 0:
            no_rain_text = alt.Chart(pd.DataFrame({'x': [predictions_df['date'].iloc[len(predictions_df)//2]], 'y': [rain_y_max/2]})).mark_text(
                text="No rain in forecast",
                color='gray',
                size=18,
                fontWeight='bold'
            ).encode(
                x='x:T',
                y='y:Q'
            ).properties(height=150)
            rain = rain + no_rain_text

        # Stack them vertically, share the X-axis, and remove outer borders.
        # View config must be on the vconcat chart, not on subcharts.
        forecast_chart = alt.vconcat(
            porcini, temp, rain, spacing=10
        ).resolve_scale(
            x='shared' # This forces them to align perfectly
        ).configure_view(
            fill='white',
            strokeOpacity=0 # Removes the box around the charts
        )

        st.altair_chart(forecast_chart, width='stretch')
    # =========================================================
    # END FORECAST CHART
    # =========================================================


    st.divider()


with tab3:
    @st.cache_data(ttl=86400) # Cache for 24 hours
    def load_historical_data():
        # A. Load Point Reyes boundary (Keep in EPSG:4326 for PyDeck)
        cpad = gpd.read_file("../Data/CPAD_Release_2025b/CPAD_2025b_Units/CPAD_2025b_Units.shp")
        pt_reyes = cpad[cpad['UNIT_NAME'] == 'Point Reyes National Seashore'].copy()
        pt_reyes_boundary = pt_reyes.dissolve().to_crs(epsg=4326)

        # B. Load and filter Vegetation Data
        # Removed the ?t=timestamp cache buster so Streamlit can cache this properly
        gcs_url_veg = "https://storage.googleapis.com/point-reyes-mushroom-data/marin_finescale_veg_dissolved.gpkg"
        veg_gpkg = gpd.read_file(gcs_url_veg).to_crs(epsg=4326)
        
        simplified_veg = veg_gpkg[veg_gpkg['ABBRV'].isin(['PiMu', 'PsMe'])].copy()
        # Simplify geometry (degrees in 4326: 0.0005 is roughly 50 meters)
        simplified_veg['geometry'] = simplified_veg['geometry'].simplify(0.00001, preserve_topology=False)
        # Clip to boundary
        simplified_veg = gpd.clip(simplified_veg, pt_reyes_boundary)

        # C. Load and filter iNaturalist Data
        inat_df = pd.read_csv('../Data/inaturalist_data/fungi/observations-675943.csv/observations-675943.csv')
        inat_df['observed_on'] = pd.to_datetime(inat_df['observed_on'])
        inat_df = inat_df[(inat_df['observed_on'].dt.year >= 2016) & (inat_df['observed_on'].dt.year <= 2024)].copy()
        inat_df['month'] = inat_df['observed_on'].dt.month
        
        CHOICE_EDIBLES = {'King Bolete': ['Boletus edulis', 'Boletus edulis var. grandedulis', 'Boletus edulis grandedulis']}
        PROXY_SPECIES = {'King Bolete': ['Suillus', 'Amanita muscaria']}
        
        all_targets = set(sp.strip().lower() for lst in {**CHOICE_EDIBLES, **PROXY_SPECIES}.values() for sp in lst)
        
        filtered_df = inat_df[inat_df['scientific_name'].str.strip().str.lower().apply(
            lambda sci: any(target in sci for target in all_targets)
        )].copy()

        edibles_gdf = gpd.GeoDataFrame(
            filtered_df,
            geometry=gpd.points_from_xy(filtered_df.longitude, filtered_df.latitude),
            crs="EPSG:4326"
        )
        # Clip observations to Point Reyes
        edibles_gdf = gpd.clip(edibles_gdf, pt_reyes_boundary)

        return edibles_gdf, simplified_veg

    
    edibles_gdf, simplified_veg_gpkg = load_historical_data()

    # Build the base Folium Map
    # @st.cache_data(ttl=86400) # Cache for 24 hours
    # def build_base_map():
    m = folium.Map(location=[38.05, -122.85], zoom_start=11)

    # Add the vegetation layers with the simplified geometries
    pimu_layer = folium.GeoJson(
        simplified_veg_gpkg[simplified_veg_gpkg.ABBRV == 'PiMu'], 
        name='Bishop Pine',
        style_function=lambda x: {
            'fillColor': '#228B22',
            'color': '#228B22',
            'weight': 1,
            'fillOpacity': 0.6
        }
    ).add_to(m)
    psme_layer = folium.GeoJson(
        simplified_veg_gpkg[simplified_veg_gpkg.ABBRV == 'PsMe'], 
        name='Douglas Fir',
        style_function=lambda x: {
            'fillColor': '#00008B',
            'color': '#00008B',
            'weight': 1,
            'fillOpacity': 0.6
        }
    ).add_to(m)

    # return m

    # Create interactive historical data map and chart
    # ==================================================
    # m = build_base_map()

    st.header("Where and When to Find Porcini")

    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("images/baby.jpg", width='stretch')
    with col2:
        st.text("""
            Porcini mushrooms in Point Reyes are most commonly found during the months of October through February, first appearing in the weeks following the first soaking rain of the season.
        
            Porcini mushrooms are also mycorrhizal, meaning they grow in relationship with trees, sharing nutrients and water. In Point Reyes, they are most commonly found in association with Bishop Pine *Pinus muricata*, though they are also often found with Douglas Fir *Pseudotsuga menziesii*.

            Explore the historical data compiled below, containing all observations of porcini and associated species (often referred to as indicator species). Bishop Pine forest, where you will find the majority of sightings, is shaded green. Douglas Fir forest is blue.

            Note that sightings are clustered around trails - more mushroom observations are reported where there are more people! Conversely, areas with few reported sightings may simply have lighter foot traffic. Don't assume that you are most likely to see porcini in the most popular areas. Sometimes it is best to get off the beaten track!        
            """)

    # Create the Widget
    month_dict = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    options = list(month_dict.keys())
    selection = st.pills("Months", options, selection_mode="multi", default=["Oct", "Nov", "Dec"])

    # Convert selection to integers
    selection_int = [month_dict[month] for month in selection]

    # Filter the mushroom data
    filtered_edibles_gdf = edibles_gdf[edibles_gdf['month'].isin(selection_int)]
    
    # Add points for each mushroom sighting
    marker_color = '#8B4513'
    fill_bool = True
    fill_op = 1.0
    for idx, row in filtered_edibles_gdf.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=marker_color,
            fill=fill_bool,
            fill_color=None,
            weight=2, # Edge thickness
            fill_opacity=0.0 #fill_op
        ).add_to(m)


    

    # Plot the timeline of sightings
    timeline_chart = alt.Chart(filtered_edibles_gdf).mark_bar().encode(
        x=alt.X('monthdate(observed_on):O', title='Month/Day (MM/DD)'),
        y=alt.Y('count()', title='Count')
    )
    st.altair_chart(timeline_chart, use_container_width=True, height=250)
    # Plot the map
    st_folium(m, use_container_width=True, height=600)


    # col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    # with col1:  
    #     st.image("images/big_porcini.jpg", width=200)
    # with col2:
    #     st.image("images/underside.jpg", width=200)
    # with col3:
    #     st.image("images/in_ground.jpg", width=200)
    # with col4:
    #     st.image("images/baby.jpg", width=200)

with tab4:
    st.header("Contact")
    st.write("Created by **Ben Yeager**")
    # st.write("Tools: Python, GeoPandas, Folium, Streamlit")
    st.write("**Email**: yeager.ben.a@gmail.com")
    st.write("**LinkedIn**: [Ben Yeager](https://www.linkedin.com/in/ben-yeager/)")
    st.write("**GitHub**: [Porcini Predictor Project](https://github.com/yeagerba/porcini_predictor)")
    # st.button("Contact Me") # Link to LinkedIn

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:  
        st.image("images/big_porcini.jpg", width=200)
    with col2:
        st.image("images/underside.jpg", width=200)
    with col3:
        st.image("images/in_ground.jpg", width=200)
    with col4:
        st.image("images/baby.jpg", width=200)

# --- TAB 3: METHODOLOGY ---
# with tab3:
#     st.header("How this Model Works")
#     st.markdown("""
#     ### 1. Spatial Component: Defining the "Where"
#     Unlike standard distribution models, this project uses **Ecological Filtering**:
#     * **Data Source:** Switched from coarse CALVEG data to the **2018 Marin Fine Scale Vegetation Map** (LIDAR-derived) for canopy-level precision.
#     * **Host Filtering:** Strictly isolated *Pinus muricata* (Bishop Pine) and *Pseudotsuga menziesii* (Douglas Fir).
#     * **Disturbance Layer:** Integrated **2020 Woodward Fire** perimeter data to exclude burned/dead canopy areas from the prediction set.
    
#     ### 2. Temporal Component: Defining the "When"
#     *(To be populated after we build your Logistic Regression model)*
#     """)

# # --- SIDEBAR ---
# with st.sidebar:
#     st.subheader("Project Info")
#     st.write("Created by **Ben Yeager**")
#     st.write("Tools: Python, GeoPandas, Folium, Streamlit")
#     st.button("Contact Me") # Link to LinkedIn