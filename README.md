
# 🔥 Global FireTracker 2024

**Track, Analyze, and Predict Global Wildfires using MODIS 2024 Data**

![FireTracker Banner](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Wildfire_icon.svg/512px-Wildfire_icon.svg.png)

---

## 📌 Overview

**Global FireTracker** is an interactive dashboard built with **Streamlit** to:
- Visualize global fire hotspots (MODIS 2024).
- Analyze fire characteristics across countries.
- Predict fire behavior using machine learning models.
- Support both public users and data scientists in understanding wildfire patterns and trends.

The dashboard supports multi-country data visualization, analytics, and modeling with a user-friendly UI.

---

## 📁 Dataset

The app uses **NASA FIRMS MODIS 2024** CSV files stored in:

```bash
modis_2024_all_countries/
    ├── modis_2024_Egypt.csv
    ├── modis_2024_Brazil.csv
    ├── modis_2024_India.csv
    └── ...etc
```

Each file contains satellite-detected fire records with fields like:
- `latitude`, `longitude`
- `acq_date`, `acq_time`
- `brightness`, `frp`, `confidence`
- `satellite`, `daynight`

---

## 🧠 Key Features

### 🌍 1. **Map View Tab**
Visualize fire locations using two interactive maps:
- **Heatmap**: Intensity-based fire detection.
- **Cluster**: Popups showing fire FRP and confidence.

Filters:
- Date range
- Confidence levels (`High`, `Medium`, `Low`)
- Satellites (`Aqua`, `Terra`)

---

### 📈 2. **Analytics Tab**
Visual insights using Plotly charts:
- 🔄 Temporal distribution: fires by Hour, Day, Week, or Month.
- 🔥 Brightness vs FRP scatter.
- 📊 Confidence level pie chart.
- 📡 Satellite comparisons for FRP and Confidence.
- 🚀 Top 10 FRP fires visualization.

---

### 🤖 3. **ML Modeling Tab**

Supports **two user modes**:
#### 🌍 Public User
Quick prediction tools for non-experts:
- 🔆 Is the fire during **Day or Night**?
- 🔥 Estimate **Brightness** (Kelvin)

Inputs:
- Country, Month, Hour, Day of Week, FRP

#### 🔬 Data Scientist
Advanced tools for training/testing models:
- Classification or Regression tasks
- Choose ML model (Random Forest, Logistic, XGBoost, etc.)
- Select features and targets
- Train/test split evaluation
- 📊 Feature importance visualization
- 📌 Custom input prediction

Optional: Simulated model saving/deployment.

---

## ⚙️ Technologies Used

| Library        | Purpose                               |
|----------------|----------------------------------------|
| `streamlit`    | Web UI for dashboard                   |
| `folium`       | Map rendering (heatmap, markers)       |
| `plotly`       | Interactive charts                     |
| `pandas`       | Data manipulation                      |
| `scikit-learn` | Machine learning models                |
| `xgboost`      | Advanced ML model (XGBoost)            |
| `seaborn` / `matplotlib` | Feature importance plots     |

---

## 🏁 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/firetracker-2024.git
cd firetracker-2024
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place your MODIS 2024 fire data in:

```
modis_2024_all_countries/
```

File naming format: `modis_2024_<Country>.csv`

### 4. Run the App

```bash
streamlit run firetracker_app.py
```

---

## 📁 Folder Structure

```
firetracker-2024/
│
├── firetracker_app.py         # 🔥 Main Streamlit app
├── modis_2024_all_countries/  # 📁 Folder with fire CSVs
├── README.md                  # 📘 Documentation
└── requirements.txt           # 📦 Python dependencies
```

---

## 🧪 Sample Use Cases

- **Disaster Management Teams**: Quickly locate hotspots and assess fire intensity.
- **Environmental Analysts**: Monitor fire trends and satellite coverage.
- **Researchers**: Build and evaluate ML models for fire prediction.

---

## 📌 Future Ideas

- Support for **VIIRS** satellite data
- Export trained models (.pkl)
- Real-time alerts or notifications
- Integration with weather APIs
- Fire risk prediction for upcoming dates

---

## 📄 License

This project is for academic and educational use only. NASA FIRMS data is publicly available.

---

## 🤝 Acknowledgements

- 🔭 **NASA FIRMS** for the MODIS fire datasets.
- 🌐 Built using **Streamlit**, **Scikit-Learn**, **XGBoost**, **Folium**, and **Plotly**.
