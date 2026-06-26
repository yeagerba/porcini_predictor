import streamlit as st
from streamlit.components.v1 import html
import os
import pandas as pd
import altair as alt
import datetime as dt
import json

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
tab1, tab2 = st.tabs(["Forecast", "Where to Look"])

# --- TAB 1: FORECAST ---
with tab1:
    st.header("OVERVIEW")
    # st.info("This section will host the Temporal Model.")
    
    # st.markdown("""
    # **Upcoming Logic:**
    # 1. **Fetch Live Weather:** Pulling daily rain/temp from NOAA for Point Reyes Station.
    # 2. **Calculate Features:** * *30-Day Cumulative Rain*
    #     * *Soil Temp Shock (7-day delta)*
    # 3. **Logistic Regression:** Probability Score (0-100%).
    # """)

    # Load the predictions
    predictions_df = pd.read_csv('predictions.csv')
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
    infotext = "Likelihood of porcini fruiting is currently "
    for key, value in likelihood_dict.items():
        if predictions_df['porcini_index'].iloc[0] <= key:
            infotext += value
            current_likelihood = value
            break
    infotext += ".\n\n"
    infotext += "Maximum likelihood of porcini fruiting within the next 14 days is "
    for key, value in likelihood_dict.items():
        if predictions_df['porcini_index'].max() <= key:
            infotext += value
            forecast_likelihood = value
            break
    infotext += ".\n\n"

    if current_likelihood == 'LOW' and forecast_likelihood == 'LOW' and predictions_df['date'].iloc[0].month in [4, 5, 6, 7, 8, 9]:
        infotext = "**Porcini season in Point Reyes is typically from October to February. Check back later!**\n\n" + infotext
   
    
    st.info(infotext)


    st.header("FORECAST")

    # # Load the forecast chart from the JSON file
    # with open('visualizations/forecast_chart.json', 'r') as f:
    #     chart_dict = json.load(f)
    
    # # Remove any hardcoded top-level width
    # chart_dict.pop('width', None)
    
    # # Remove hidden config-level widths that override container settings
    # if 'config' in chart_dict and 'view' in chart_dict['config']:
    #     chart_dict['config']['view'].pop('continuousWidth', None)
    #     chart_dict['config']['view'].pop('width', None)
        
    # # Prevent padding from pushing the container width past 100%
    # chart_dict['padding'] = {
    #     "left": 10,
    #     "right": 1000, # Increase this if the right side is still spilling over
    #     "top": 10,
    #     "bottom": 10
    # }
    # chart_dict['autosize'] = {"type": "fit", "contains": "padding"}
    
    # # Apply responsive width to the individual sub-charts
    # if 'vconcat' in chart_dict:
    #     for subchart in chart_dict['vconcat']:
    #         subchart['width'] = 'container'

    # st.vega_lite_chart(chart_dict, width='stretch') #, theme='streamlit')


    # ================================================
    # FORECAST CHART
    # ================================================
    predictions_df = pd.read_csv('predictions.csv')
    # Create a base layer with the shared X-axis (formatted as Day-Date, no vertical grid)
    base = alt.Chart(predictions_df).encode(
        x=alt.X('date:T', axis=alt.Axis(format='%a %d', title=None, grid=False, labelAngle=-45, tickCount=14))
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
        # width=alt.Step(30),
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
        # width=alt.Step(30),
        height=150
    )

    porcini_line = base.mark_line(
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

    porcini_points = base.mark_point(
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
        height=100
    )

    temp = (
        temp_grid_chart
        + base.mark_line(
            color='#FF4B4B', strokeWidth=1.5, interpolate='monotone'
        ).encode(
            y=alt.Y(
                'tmax_c_true:Q',
                title='Temp (°C)',
                scale=alt.Scale(domain=[0, 25]),
                axis=alt.Axis(grid=False)
            )
        ).properties(height=100)
        + base.mark_point(
            color='#FF4B4B', filled=True, size=30
        ).encode(
            y=alt.Y('tmax_c_true:Q', scale=alt.Scale(domain=[0, 25]))
        ).properties(height=100)
        + base.mark_line(
            color='#0077B6', strokeWidth=1.5, interpolate='monotone'
        ).encode(
            y=alt.Y(
                'tmin_c_true:Q',
                title='Temp (°C)',
                scale=alt.Scale(domain=[0, 25]),
                axis=alt.Axis(grid=False)
            )
        ).properties(height=100)
        + base.mark_point(
            color='#0077B6', filled=True, size=30
        ).encode(
            y=alt.Y('tmin_c_true:Q', scale=alt.Scale(domain=[0, 25]))
        ).properties(height=100)
    )

    # Set rain y-axis domain: [0, max(10, prcp_mm_true.max())]
    rain_y_max = max(10, predictions_df['prcp_mm_true'].max())
    rain = base.mark_bar(
        color='#3A86FF', opacity=0.8
    ).encode(
        y=alt.Y(
            'prcp_mm_true:Q',
            title='Rain (mm)',
            scale=alt.Scale(domain=[0, rain_y_max]),
            axis=alt.Axis(grid=False)
        )
    ).properties(height=100)

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
        ).properties(height=100)
        rain = rain + no_rain_text

    # Stack them vertically, share the X-axis, and remove outer borders
    forecast_chart = alt.vconcat(
        porcini, temp, rain, spacing=10
    ).resolve_scale(
        x='shared' # This forces them to align perfectly
    ).configure_view(
        strokeOpacity=0 # Removes the box around the charts
    )

    st.altair_chart(forecast_chart, width='stretch')
    # =========================================================
    # END FORECAST CHART
    # =========================================================


    st.divider()

# --- TAB 2: WHERE TO LOOK ---
with tab2:
    st.header("Where to find porcini")

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

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("Project Info")
    st.write("Created by **Ben Yeager**")
    st.write("Tools: Python, GeoPandas, Folium, Streamlit")
    st.button("Contact Me") # Link to LinkedIn