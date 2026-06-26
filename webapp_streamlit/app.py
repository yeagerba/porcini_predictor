import streamlit as st
from streamlit.components.v1 import html
import os
import pandas as pd
import altair as alt
import datetime as dt
import json
import time

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
tab1, tab2, tab3 = st.tabs(["Forecast", "About Porcini & Where to Find Them", "Contact"])

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
        infotext = "**No porcini expected in forecast period. Porcini season in Point Reyes runs from Oct-Feb. Check back later!**"
   
    
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
                title='Fruiting Likelihood',
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(
                    grid=False,
                    values=yticks,
                    tickCount=len(yticks),
                    ticks=True,
                    labels=True,
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

with tab2:
    st.header("About Porcini")

    col1, col2 = st.columns([2, 4])

    with col1:
        st.image("images/big_porcini.jpg", width=200)
        st.image("images/underside.jpg", width=200)
        st.image("images/in_ground.jpg", width=200)
        st.image("images/baby.jpg", width=200)
    with col2:
        st.text("""
        Porcini mushrooms are mycorrhizal, meaning they grow in relationship with trees, sharing nutrients and water. 
        In Point Reyes, they are most commonly found in association with Bishop Pine *Pinus muricata*, though they are also often found with Douglas Fir *Pseudotsuga menziesii*.

        The map below shows the locations of Bishop Pine (in green) and Douglas Fir (in blue) in Point Reyes. 
        Dotted lines represent trails - find a trail that runs through Bishop Pine habitat for your best chance to find porcini!

        Note also the red region denoting the area affected by the 2020 Woodward Fire. The forest in this area is still recovering and has large areas of thick undergrowth that can make it difficult to find porcini.
        """)

        # Load the map HTML file
        with open("visualizations/vegetation_trail_map.html", "r", encoding="utf-8") as f:
            map_html = f.read()

        # Display the map using Streamlit's HTML component
        html(map_html, height=600, width=1000)

with tab3:
    st.header("Contact")
    st.write("Created by **Ben Yeager**")
    # st.write("Tools: Python, GeoPandas, Folium, Streamlit")
    st.write("**Email**: yeager.ben.a@gmail.com")
    st.write("**LinkedIn**: [Ben Yeager](https://www.linkedin.com/in/ben-yeager/)")
    st.write("**GitHub**: [Porcini Predictor Project](https://github.com/yeagerba/porcini_predictor)")
    # st.button("Contact Me") # Link to LinkedIn

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