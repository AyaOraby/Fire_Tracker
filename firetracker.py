import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static, st_folium
import plotly.express as px
from datetime import datetime, timedelta
import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Global FireTracker",
    page_icon="🔥",
    layout="wide"
)

# Constants
DATA_FOLDER = "modis_2024_all_countries"

# Country center coordinates (latitude, longitude, zoom level)
COUNTRY_CENTERS = {
    "Egypt": (26.8206, 30.8025, 6),
    "USA": (37.0902, -95.7129, 4),
    "Australia": (-25.2744, 133.7751, 4),
    "Brazil": (-14.2350, -51.9253, 4),
    "India": (20.5937, 78.9629, 5),
    "Canada": (56.1304, -106.3468, 3),
    "Russia": (61.5240, 105.3188, 3),
    "China": (35.8617, 104.1954, 4),
    "Indonesia": (-0.7893, 113.9213, 5),
    "Mexico": (23.6345, -102.5528, 5),
    "Vietnam": (14.0583, 108.2772, 6)
}

# Function to get available countries
def get_available_countries():
    try:
        files = glob.glob(os.path.join(DATA_FOLDER, "modis_2024_*.csv"))
        if not files:
            st.error(f"No data files found in {DATA_FOLDER} folder. Please check your data files.")
            return []
            
        countries = [os.path.basename(f).split('_')[2].replace(".csv", "") for f in files]
        return sorted(countries)
    except Exception as e:
        st.error(f"Error scanning data folder: {str(e)}")
        return []

# Load data function with caching
@st.cache_data
def load_data(country):
    try:
        file_path = os.path.join(DATA_FOLDER, f"modis_2024_{country}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found for {country}")
            
        df = pd.read_csv(file_path)
        
        # Data preprocessing
        df['acq_datetime'] = pd.to_datetime(
            df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4),
            format='%Y-%m-%d %H%M'
        )
        df['confidence_category'] = df['confidence'].apply(
            lambda x: 'High' if x >= 80 else ('Medium' if 50 <= x < 80 else 'Low')
        )
        
        # Create temporal features
        df['day_of_year'] = df['acq_datetime'].dt.dayofyear
        df['week_of_year'] = df['acq_datetime'].dt.isocalendar().week
        df['month'] = df['acq_datetime'].dt.month
        df['hour'] = df['acq_datetime'].dt.hour
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data for {country}: {str(e)}")
        return None

# Get available countries
available_countries = get_available_countries()

if not available_countries:
    st.error("No country data available. Please ensure your data files are properly named and placed in the modis_2024_all_countries folder.")
    st.stop()

# Sidebar - Country Selection
st.sidebar.header("Country Selection")
selected_country = st.sidebar.selectbox(
    "Select Country",
    options=available_countries,
    index=0
)

# Load data for selected country
df = load_data(selected_country)

if df is None or df.empty:
    st.error(f"No valid data available for {selected_country}. Please check your data file.")
    st.stop()

# Get map center for selected country
map_center = COUNTRY_CENTERS.get(selected_country, (df['latitude'].mean(), df['longitude'].mean(), 5))

# Dashboard Title
st.title(f"🔥 FireTracker - {selected_country}")
st.markdown(f"""
Tracking wildfire hotspots and environmental impact in {selected_country} (2024)
""")

# Sidebar Filters
st.sidebar.header("Filter Options")

# Get min and max dates from data
min_date = df['acq_datetime'].min().date()
max_date = df['acq_datetime'].max().date()

# Separate 'From' and 'To' date inputs
from_date = st.sidebar.date_input(
    "From",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    key="from_date"
)

to_date = st.sidebar.date_input(
    "To",
    value=max_date,
    min_value=from_date,  # prevent selecting to_date before from_date
    max_value=max_date,
    key="to_date"
)

# Confidence level filter
confidence_levels = st.sidebar.multiselect(
    "Confidence Levels",
    options=['High', 'Medium', 'Low'],
    default=['High', 'Medium', 'Low']
)

# Satellite filter
satellites = st.sidebar.multiselect(
    "Satellites",
    options=df['satellite'].unique(),
    default=df['satellite'].unique()
)

# Apply filters
filtered_df = df[
    (df['acq_datetime'].dt.date >= from_date) &
    (df['acq_datetime'].dt.date <= to_date) &
    (df['confidence_category'].isin(confidence_levels)) &
    (df['satellite'].isin(satellites))
]

# Main Dashboard Layout
tab1, tab2 ,tab4 = st.tabs(["🌍 Map View", "📈 Analytics", "ML Modeling"])

with tab1:
    st.header("Fire Hotspots")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        map_type = st.radio(
            "Map Type",
            ["Heatmap", "Cluster"],
            horizontal=True
        )

        m = folium.Map(location=[map_center[0], map_center[1]], zoom_start=map_center[2])

        if map_type == "Heatmap":
            heat_data = [[row['latitude'], row['longitude']] for _, row in filtered_df.iterrows()]
            HeatMap(heat_data, radius=15).add_to(m)

        elif map_type == "Cluster":
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in filtered_df.iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"FRP: {row['frp']} MW<br>Confidence: {row['confidence']}%",
                    icon=folium.Icon(color='red', icon='fire')
                ).add_to(marker_cluster)

        folium_static(m, width=800, height=600)
    
    with col2:
        st.metric("Total Fires Detected", len(filtered_df))
        st.metric("Average FRP (MW)", round(filtered_df['frp'].mean(), 2))
        st.metric("Average Confidence", f"{round(filtered_df['confidence'].mean(), 2)}%")
        
        st.download_button(
            label="Download Filtered Data",
            data=filtered_df.to_csv(index=False),
            file_name=f"filtered_fire_data_{selected_country}.csv",
            mime="text/csv"
        )

with tab2:
    st.header("📊 Analytical Views")

    col1, col2 = st.columns(2)

    # --- Column 1: Temporal Analysis ---
    with col1:
        st.subheader("📅 Temporal Analysis")
        time_agg = st.selectbox(
            "Aggregate fires by:",
            ["Hour", "Day", "Week", "Month"]
        )

        if time_agg == "Hour":
            time_df = filtered_df.groupby(filtered_df['acq_datetime'].dt.hour).size()
            time_df.index = time_df.index.astype(str) + ":00"
        elif time_agg == "Day":
            time_df = filtered_df.groupby(filtered_df['acq_datetime'].dt.day).size()
        elif time_agg == "Week":
            time_df = filtered_df.groupby(filtered_df['acq_datetime'].dt.isocalendar().week).size()
        else:
            time_df = filtered_df.groupby(filtered_df['acq_datetime'].dt.month).size()

        fig = px.bar(
            time_df,
            labels={'value': 'Fire Count', 'index': time_agg},
            title=f"🔥 Fires by {time_agg}",
            color_discrete_sequence=['indianred']
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Column 2: Fire Characteristics ---
    with col2:
        st.subheader("🔥 Fire Characteristics")
        char_option = st.selectbox(
            "Select view:",
            ["Brightness vs FRP", "Confidence Distribution", "Satellite Comparison", "Top Fires"]
        )

        if char_option == "Brightness vs FRP":
            fig = px.scatter(
                filtered_df,
                x='brightness',
                y='frp',
                color='confidence_category',
                size='frp',
                title="🔥 FRP vs Brightness",
                labels={"brightness": "Brightness", "frp": "FRP (MW)"},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig, use_container_width=True)

        elif char_option == "Confidence Distribution":
            fig = px.pie(
                filtered_df,
                names='confidence_category',
                title="🔥 Confidence Level Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

        elif char_option == "Satellite Comparison":
            st.markdown("### 📡 Satellite FRP & Confidence Comparison")

            fig1 = px.box(
                filtered_df,
                x='satellite',
                y='frp',
                color='satellite',
                title="FRP Distribution by Satellite",
                labels={'frp': 'FRP (MW)', 'satellite': 'Satellite'},
                color_discrete_map={"Aqua": "royalblue", "Terra": "darkorange"}
            )
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.histogram(
                filtered_df,
                x='confidence',
                color='satellite',
                barmode='overlay',
                nbins=20,
                title="📶 Confidence Levels by Satellite",
                color_discrete_map={"Aqua": "royalblue", "Terra": "darkorange"}
            )
            st.plotly_chart(fig2, use_container_width=True)

        elif char_option == "Top Fires":
            st.markdown("### 🔥 Top 10 Fires by Fire Radiative Power (FRP)")

            top_fires = filtered_df.nlargest(10, 'frp')
            top_fires['acq_time_formatted'] = top_fires['acq_datetime'].dt.strftime('%Y-%m-%d %H:%M')

            fig = px.scatter(
                top_fires,
                x='acq_datetime',
                y='frp',
                color='confidence_category',
                size='brightness',
                hover_data={
                    'acq_datetime': True,
                    'frp': ':.2f',
                    'brightness': ':.1f',
                    'latitude': ':.2f',
                    'longitude': ':.2f',
                    'confidence_category': True
                },
                title="Top 10 Fires by FRP",
                labels={'acq_datetime': 'Date & Time', 'frp': 'FRP (MW)'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )

            fig.update_layout(
                xaxis_title="Date & Time",
                yaxis_title="FRP (MW)",
                hoverlabel=dict(bgcolor="white", font_size=12),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)



with tab4:

    # ⚙️ Page Setup
    st.set_page_config(page_title="🔥 Fire Prediction - MODIS 2024", layout="wide")

    DATA_FOLDER = "modis_2024_all_countries"

    COUNTRY_CENTERS = {
        "Egypt": (26.8206, 30.8025, 6),
        "USA": (37.0902, -95.7129, 4),
        "Australia": (-25.2744, 133.7751, 4),
        "Brazil": (-14.2350, -51.9253, 4),
        "India": (20.5937, 78.9629, 5),
        "Canada": (56.1304, -106.3468, 3),
        "Russia": (61.5240, 105.3188, 3),
        "China": (35.8617, 104.1954, 4),
        "Indonesia": (-0.7893, 113.9213, 5),
        "Mexico": (23.6345, -102.5528, 5),
        "Vietnam": (14.0583, 108.2772, 6),
        "Albania": (41.1533, 20.1683, 7)
    }

    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    MONTH_NAMES = ["January", "February", "March", "April", "May", "June", 
                "July", "August", "September", "October", "November", "December"]

    FEATURE_INFO = {
        'latitude': {'display': 'Latitude', 'description': 'Geographic coordinate specifying north-south position', 'min': -90, 'max': 90, 'type': 'float'},
        'longitude': {'display': 'Longitude', 'description': 'Geographic coordinate specifying east-west position', 'min': -180, 'max': 180, 'type': 'float'},
        'brightness': {'display': 'Brightness Temperature', 'description': 'Temperature of the fire (Kelvin scale)', 'min': 300, 'max': 500, 'type': 'float'},
        'bright_t31': {'display': 'Channel 31 Brightness', 'description': 'Brightness temperature in MODIS channel 31 (Kelvin)', 'min': 250, 'max': 350, 'type': 'float'},
        'frp': {'display': 'Fire Radiative Power', 'description': 'Energy emitted by fire (MW)', 'min': 0, 'max': 1000, 'type': 'float'},
        'day_of_week': {'display': 'Day of Week', 'description': 'Day of the week', 'min': 0, 'max': 6, 'type': 'int', 'options': DAY_NAMES},
        'hour': {'display': 'Hour of Day', 'description': 'Time of observation (0-23)', 'min': 0, 'max': 23, 'type': 'int'},
        'month': {'display': 'Month', 'description': 'Month of the year', 'min': 1, 'max': 12, 'type': 'int', 'options': MONTH_NAMES}
    }

    def get_available_countries():
        try:
            files = glob.glob(os.path.join(DATA_FOLDER, "modis_2024_*.csv"))
            if not files:
                st.error("No data files found.")
                return []
            countries = [os.path.basename(f).split('_')[2].replace(".csv", "") for f in files]
            return sorted(countries)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return []

    @st.cache_data
    def load_data(country):
        path = os.path.join(DATA_FOLDER, f"modis_2024_{country}.csv")
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        df['acq_datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4))
        df['day_of_week'] = df['acq_datetime'].dt.dayofweek
        df['hour'] = df['acq_datetime'].dt.hour
        df['month'] = df['acq_datetime'].dt.month
        df.drop(columns=['acq_date', 'acq_time', 'acq_datetime'], inplace=True)
        return df

    # ────────────────────────────────
    # USER MODE SELECTION
    # ────────────────────────────────
    st.markdown("## 🔥 Fire Prediction Dashboard")
    user_mode = st.radio("Select User Mode:", ["🌍 Public User", "🔬 Data Scientist"], horizontal=True)

    if user_mode == "🌍 Public User":
        with st.expander("🧪 Simple Fire Prediction (Public User)", expanded=True):
            available_countries = get_available_countries()
            selected_country = st.selectbox("🌍 Select Country", available_countries)

            if selected_country:
                df = load_data(selected_country)
                st.subheader("🧪 Simple Prediction Tool")
                task = st.radio("Prediction Type", ["Day or Night?", "Brightness Estimation"])

                month = st.selectbox(FEATURE_INFO['month']['display'],
                                     options=range(FEATURE_INFO['month']['min'], FEATURE_INFO['month']['max']+1),
                                     format_func=lambda x: MONTH_NAMES[x-1],
                                     index=6, help=FEATURE_INFO['month']['description'])

                hour = st.slider(FEATURE_INFO['hour']['display'],
                                 min_value=FEATURE_INFO['hour']['min'],
                                 max_value=FEATURE_INFO['hour']['max'],
                                 value=13, help=FEATURE_INFO['hour']['description'])

                day_of_week = st.selectbox(FEATURE_INFO['day_of_week']['display'],
                                           options=range(FEATURE_INFO['day_of_week']['min'], FEATURE_INFO['day_of_week']['max']+1),
                                           format_func=lambda x: DAY_NAMES[x], index=3,
                                           help=FEATURE_INFO['day_of_week']['description'])

                frp = st.slider(FEATURE_INFO['frp']['display'],
                                min_value=float(FEATURE_INFO['frp']['min']),
                                max_value=float(FEATURE_INFO['frp']['max']),
                                value=30.0, help=FEATURE_INFO['frp']['description'])

                brightness = 350.0
                bright_t31 = 310.0
                lat, lon = COUNTRY_CENTERS.get(selected_country, (0.0, 0.0))[:2]

                input_data = pd.DataFrame([{
                    'latitude': lat,
                    'longitude': lon,
                    'hour': hour,
                    'month': month,
                    'frp': frp,
                    'brightness': brightness,
                    'bright_t31': bright_t31,
                    'day_of_week': day_of_week
                }])

                if st.button("🔍 Predict"):
                    if task == "Day or Night?":
                        y = df['daynight'].map({'D': 0, 'N': 1})
                        X = df[list(FEATURE_INFO.keys())]
                        model = RandomForestClassifier()
                        model.fit(X, y)
                        prediction = model.predict(input_data[list(FEATURE_INFO.keys())])[0]
                        label = "🌞 Day" if prediction == 0 else "🌙 Night"
                        st.success(f"Prediction: {label}")
                        st.write(f"**Context:**\n- Day of Week: {DAY_NAMES[day_of_week]}\n- Month: {MONTH_NAMES[month-1]}\n- Hour: {hour}:00\n- Country: {selected_country}")
                    else:
                        y = df['brightness']
                        X = df[list(FEATURE_INFO.keys())]
                        model = RandomForestRegressor()
                        model.fit(X, y)
                        prediction = model.predict(input_data[list(FEATURE_INFO.keys())])[0]
                        st.success(f"Predicted Brightness: {prediction:.2f} Kelvin")

    elif user_mode == "🔬 Data Scientist":
        with st.expander("🔬 Model Testing & Deployment (Data Scientist)", expanded=True):
            mode = st.radio("Choose Mode:", ["General Training", "Train with Selected Features"])
            available_countries = get_available_countries()
            selected_country = st.selectbox("🌍 Choose Dataset Country", available_countries, key="ds_country")

            if selected_country:
                df = load_data(selected_country)
                st.subheader("🧠 Advanced Model Training & Evaluation")
                task_type = st.radio("Task Type", ["Classification", "Regression"])

                if task_type == "Classification":
                    target_col = 'daynight'
                    y = df[target_col].map({'D': 0, 'N': 1})
                    model_name = st.selectbox("Model", ["Random Forest", "Logistic Regression", "XGBoost"])
                else:
                    target_col = st.selectbox("Choose Target", ['brightness', 'frp', 'bright_t31'])
                    y = df[target_col]
                    model_name = st.selectbox("Model", ["Random Forest Regressor", "Linear Regression", "XGBoost Regressor"])

                if mode == "Train with Selected Features":
                    selected_features = st.multiselect("Select Features to Use", list(FEATURE_INFO.keys()), default=list(FEATURE_INFO.keys()))
                else:
                    selected_features = list(FEATURE_INFO.keys())

                X = df[selected_features]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = None
                if model_name == "Random Forest":
                    model = RandomForestClassifier()
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_name == "XGBoost":
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                elif model_name == "Random Forest Regressor":
                    model = RandomForestRegressor()
                elif model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "XGBoost Regressor":
                    model = XGBRegressor()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if task_type == "Classification":
                    st.subheader("📊 Classification Results")
                    st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred, target_names=["Day", "Night"]))
                else:
                    st.subheader("📊 Regression Results")
                    st.text(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
                    st.text(f"R² Score: {r2_score(y_test, y_pred):.2f}")

                st.subheader("📌 Feature Importance")
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = model.coef_[0]
                else:
                    importance = None

                if importance is not None:
                    display_names = [FEATURE_INFO[feat]['display'] for feat in selected_features]
                    feat_df = pd.DataFrame({'Feature': display_names, 'Importance': importance})
                    feat_df = feat_df.sort_values(by='Importance', ascending=False)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(data=feat_df, x='Importance', y='Feature', ax=ax)
                    st.pyplot(fig)

                if mode == "Train with Selected Features":
                    st.subheader("🚀 Deploy Trained Model")
                    model_name_input = st.text_input("Model Name to Save", f"{model_name}_{selected_country}")
                    if st.button("💾 Save & Deploy (Simulated)"):
                        st.success(f"Model '{model_name_input}' saved and deployed (simulated).")

                    st.subheader("🧪 Custom Prediction using Trained Model")
                    cols = st.columns(2)
                    user_input = {}
                    for i, feat in enumerate(selected_features):
                        with cols[i % 2]:
                            info = FEATURE_INFO[feat]
                            if info['type'] == 'int' and 'options' in info:
                                val = st.selectbox(info['display'],
                                                options=range(info['min'], info['max']+1),
                                                format_func=lambda x: info['options'][x] if feat == 'day_of_week' else info['options'][x-1],
                                                index=int(df[feat].mode()[0]))
                            elif info['type'] == 'int':
                                val = st.number_input(info['display'], min_value=info['min'], max_value=info['max'], value=int(df[feat].mean()))
                            else:
                                val = st.number_input(info['display'], min_value=float(info['min']), max_value=float(info['max']), value=float(df[feat].mean()))
                            user_input[feat] = val

                    if st.button("📌 Predict with Custom Input"):
                        input_df = pd.DataFrame([user_input])
                        result = model.predict(input_df)[0]
                        if task_type == "Classification":
                            if target_col == "type":
                                st.success("🔥 Prediction: Fire" if result == 1 else "🟢 Prediction: No Fire")
                            else:
                                st.success("Prediction: 🌞 Day" if result == 0 else "Prediction: 🌙 Night")
                        else:
                            st.success(f"Predicted {target_col}: {result:.2f}")


# Footer
st.markdown("---")
st.markdown("""
**Data Source**: NASA FIRMS MODIS/VIIRS Fire Data  
**Disclaimer**: Data is for informational purposes only
""")