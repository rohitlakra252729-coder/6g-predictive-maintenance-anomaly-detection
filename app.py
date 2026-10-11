
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="6G Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Predictive Maintenance & Anomaly Detection")
st.subheader("6G-Integrated Smart Manufacturing Systems")

st.markdown(
    """
    Monitor industrial machine health, detect abnormal sensor patterns,
    classify maintenance risk, and prioritize machine inspections.
    """
)

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data(file_bytes, filename):
    from io import BytesIO
    data = pd.read_csv(BytesIO(file_bytes))
    data.columns = data.columns.str.strip()
    return data


uploaded_file = st.sidebar.file_uploader(
    "Upload manufacturing dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = load_data(uploaded_file.getvalue(), uploaded_file.name)
else:
    possible_files = [
        "Thales_Group_Manufacturing.csv",
        "Manufacturing_Cleaned_Dataset.csv",
        "manufacturing_dataset.csv"
    ]

    df = None
    for filename in possible_files:
        try:
            df = pd.read_csv(filename)
            break
        except FileNotFoundError:
            continue

    if df is None:
        st.warning(
            "Upload your manufacturing CSV using the sidebar, or place "
            "the dataset in the same folder as app.py."
        )
        st.stop()

df.columns = df.columns.str.strip()

# --------------------------------------------------
# COLUMN MAPPING
# --------------------------------------------------
def find_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


aliases = {
    "machine": ["Machine_ID", "Machine ID", "Machine"],
    "date": ["Date"],
    "time": ["Time", "Timestamp"],
    "mode": ["Operation_Mode", "Operation Mode"],
    "temperature": ["Temperature_C", "Temperature", "Temperature C"],
    "vibration": ["Vibration_Hz", "Vibration", "Vibration Hz"],
    "power": ["Power_Consumption_kW", "Power Consumption", "Power"],
    "latency": ["Network_Latency_ms", "Network Latency", "Latency"],
    "packet_loss": ["Packet_Loss_%", "Packet Loss", "Packet Loss %"],
    "defect": [
        "Quality_Control_Defect_Rate_%",
        "Quality Control Defect Rate %",
        "Defect Rate"
    ],
    "speed": [
        "Production_Speed_units_per_hr",
        "Production Speed",
        "Production Speed units per hr"
    ],
    "maintenance": [
        "Predictive_Maintenance_Score",
        "Maintenance Score",
        "Predictive Maintenance Score"
    ],
    "error": ["Error_Rate_%", "Error Rate", "Error Rate %"],
    "efficiency": ["Efficiency_Status", "Efficiency Status"]
}

cols = {
    key: find_column(df.columns, candidates)
    for key, candidates in aliases.items()
}

if cols["machine"] is None:
    st.error(
        "Machine ID column not found. Available columns: "
        + ", ".join(df.columns.astype(str))
    )
    st.stop()

sensor_keys = [
    "temperature", "vibration", "power", "latency",
    "packet_loss", "defect", "speed", "maintenance", "error"
]

available_sensors = [key for key in sensor_keys if cols[key] is not None]

if len(available_sensors) < 2:
    st.error(
        "At least two supported numeric sensor columns are required. "
        "Check the dataset column names."
    )
    st.stop()

# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------
df["Machine_ID"] = df[cols["machine"]].astype(str)

if cols["date"] and cols["time"] and cols["date"] != cols["time"]:
    df["Timestamp"] = pd.to_datetime(
        df[cols["date"]].astype(str) + " " +
        df[cols["time"]].astype(str),
        errors="coerce"
    )
elif cols["time"]:
    df["Timestamp"] = pd.to_datetime(df[cols["time"]], errors="coerce")
elif cols["date"]:
    df["Timestamp"] = pd.to_datetime(df[cols["date"]], errors="coerce")
else:
    df["Timestamp"] = pd.NaT

sensor_labels = {
    "temperature": "Temperature",
    "vibration": "Vibration",
    "power": "Power Consumption",
    "latency": "Network Latency",
    "packet_loss": "Packet Loss",
    "defect": "Defect Rate",
    "speed": "Production Speed",
    "maintenance": "Maintenance Score",
    "error": "Error Rate"
}

for key in available_sensors:
    df[sensor_labels[key]] = pd.to_numeric(
        df[cols[key]], errors="coerce"
    )

numeric_sensors = [sensor_labels[key] for key in available_sensors]

if cols["mode"]:
    df["Operation_Mode"] = df[cols["mode"]].fillna("Unknown").astype(str)
else:
    df["Operation_Mode"] = "Unknown"

if cols["efficiency"]:
    df["Efficiency_Status"] = df[cols["efficiency"]].astype(str)
else:
    df["Efficiency_Status"] = "Not available"

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=numeric_sensors, how="all").copy()

if len(df) < 10:
    st.error("At least 10 usable records are needed for anomaly detection.")
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

machines = sorted(df["Machine_ID"].unique().tolist())

selected_machines = st.sidebar.multiselect(
    "Select machines",
    options=machines,
    default=machines[:min(10, len(machines))]
)

modes = sorted(df["Operation_Mode"].unique().tolist())
selected_modes = st.sidebar.multiselect(
    "Operation mode",
    options=modes,
    default=modes
)

risk_threshold = st.sidebar.slider(
    "Anomaly score threshold",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.01
)

time_window = st.sidebar.selectbox(
    "Time window",
    ["All records", "Last 24 hours", "Last 7 days", "Last 30 days"]
)

filtered = df[
    df["Machine_ID"].isin(selected_machines)
    & df["Operation_Mode"].isin(selected_modes)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

if filtered["Timestamp"].notna().any() and time_window != "All records":
    latest_time = filtered["Timestamp"].max()

    window_days = {
        "Last 24 hours": 1,
        "Last 7 days": 7,
        "Last 30 days": 30
    }[time_window]

    filtered = filtered[
        filtered["Timestamp"] >=
        latest_time - pd.Timedelta(days=window_days)
    ].copy()

if filtered.empty:
    st.warning("No records remain in the selected time window.")
    st.stop()

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
# Median imputation supports missing sensor readings.
X = filtered[numeric_sensors].copy()
X = X.fillna(X.median())

# Remove sensors that contain no usable variation.
usable_sensors = [
    col for col in X.columns
    if X[col].notna().any() and X[col].nunique() > 1
]

if not usable_sensors:
    st.error("The available sensor readings do not vary enough for modeling.")
    st.stop()

X = X[usable_sensors]
X = X.fillna(X.median()).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

contamination = st.sidebar.slider(
    "Expected anomaly proportion",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01
)

model = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42
)

model.fit(X_scaled)

# Higher score means more abnormal.
anomaly_score = -model.score_samples(X_scaled)

# Convert anomaly scores to a comparable 0–100 display scale.
score_min = anomaly_score.min()
score_max = anomaly_score.max()

if score_max > score_min:
    normalized_score = (
        (anomaly_score - score_min) / (score_max - score_min) * 100
    )
else:
    normalized_score = np.zeros(len(anomaly_score))

filtered["Anomaly_Score"] = normalized_score
filtered["Model_Anomaly"] = model.predict(X_scaled) == -1

filtered["Risk_Level"] = np.select(
    [
        filtered["Anomaly_Score"] >= risk_threshold * 100,
        filtered["Anomaly_Score"] >= risk_threshold * 70
    ],
    ["High", "Medium"],
    default="Low"
)

# Keep row-aligned sensor data for visualizations.
filtered["Record_Number"] = np.arange(1, len(filtered) + 1)

# --------------------------------------------------
# OVERVIEW KPIs
# --------------------------------------------------
st.header("1. Predictive Maintenance Overview")

total_records = len(filtered)
machine_count = filtered["Machine_ID"].nunique()
high_risk_count = filtered.loc[
    filtered["Risk_Level"] == "High", "Machine_ID"
].nunique()
anomaly_count = int(filtered["Model_Anomaly"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Records Analyzed", f"{total_records:,}")
k2.metric("Machines Monitored", f"{machine_count:,}")
k3.metric("High-Risk Machines", f"{high_risk_count:,}")
k4.metric("Model-Detected Anomalies", f"{anomaly_count:,}")

# --------------------------------------------------
# RISK DISTRIBUTION
# --------------------------------------------------
st.header("2. Machine Risk Distribution")

risk_order = ["Low", "Medium", "High"]
risk_counts = (
    filtered["Risk_Level"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
    .rename_axis("Risk Level")
    .reset_index(name="Records")
)

left, right = st.columns(2)

with left:
    fig_risk = px.pie(
        risk_counts,
        names="Risk Level",
        values="Records",
        title="Risk Distribution",
        hole=0.45
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with right:
    machine_risk = (
        filtered.groupby(["Machine_ID", "Risk_Level"])
        .size()
        .reset_index(name="Records")
    )

    fig_machine = px.bar(
        machine_risk,
        x="Machine_ID",
        y="Records",
        color="Risk_Level",
        title="Risk Records by Machine",
        barmode="stack"
    )
    st.plotly_chart(fig_machine, use_container_width=True)

# --------------------------------------------------
# MACHINE ANOMALY DASHBOARD
# --------------------------------------------------
st.header("3. Machine Anomaly Dashboard")

selected_machine = st.selectbox(
    "Inspect individual machine",
    options=sorted(filtered["Machine_ID"].unique())
)

machine_data = filtered[
    filtered["Machine_ID"] == selected_machine
].copy()

if machine_data["Timestamp"].notna().any():
    machine_data = machine_data.sort_values("Timestamp")
    x_axis = "Timestamp"
else:
    x_axis = "Record_Number"

fig_score = px.line(
    machine_data,
    x=x_axis,
    y="Anomaly_Score",
    markers=True,
    title=f"Anomaly Score Trend — {selected_machine}"
)
fig_score.add_hline(
    y=risk_threshold * 100,
    line_dash="dash",
    annotation_text="High-risk threshold"
)
st.plotly_chart(fig_score, use_container_width=True)

chosen_sensors = st.multiselect(
    "Select sensor variables",
    options=usable_sensors,
    default=usable_sensors[:min(4, len(usable_sensors))]
)

if chosen_sensors:
    sensor_long = machine_data[
        [x_axis] + chosen_sensors
    ].melt(
        id_vars=x_axis,
        var_name="Sensor",
        value_name="Reading"
    )

    fig_sensor = px.line(
        sensor_long,
        x=x_axis,
        y="Reading",
        color="Sensor",
        title="Sensor Behavior Over Time"
    )
    st.plotly_chart(fig_sensor, use_container_width=True)

# --------------------------------------------------
# SENSOR DEVIATION ANALYSIS
# --------------------------------------------------
st.header("4. Sensor Deviation Analysis")

deviation = filtered[usable_sensors].copy()

for sensor in usable_sensors:
    baseline = deviation[sensor].median()
    spread = deviation[sensor].std()

    if pd.isna(spread) or spread == 0:
        deviation[sensor] = 0.0
    else:
        deviation[sensor] = (
            (deviation[sensor] - baseline) / spread
        )

deviation["Machine_ID"] = filtered["Machine_ID"].values
deviation["Anomaly_Score"] = filtered["Anomaly_Score"].values

deviation_long = deviation.melt(
    id_vars=["Machine_ID", "Anomaly_Score"],
    value_vars=usable_sensors,
    var_name="Sensor",
    value_name="Standardized Deviation"
)

fig_deviation = px.scatter(
    deviation_long,
    x="Machine_ID",
    y="Standardized Deviation",
    color="Sensor",
    title="Sensor Deviations from Overall Baseline",
    hover_data=["Anomaly_Score"]
)
fig_deviation.add_hline(y=0, line_dash="dash")
st.plotly_chart(fig_deviation, use_container_width=True)

# --------------------------------------------------
# MAINTENANCE ALERT PANEL
# --------------------------------------------------
st.header("5. Maintenance Alert Panel")

alerts = filtered[filtered["Risk_Level"] == "High"].copy()

if alerts.empty:
    st.success("No records currently exceed the configured high-risk threshold.")
else:
    st.warning(
        f"{len(alerts)} records are classified as high risk. "
        "Prioritize inspection of the machines listed below."
    )

    alert_columns = [
        "Machine_ID", "Timestamp", "Operation_Mode",
        "Anomaly_Score", "Risk_Level"
    ] + usable_sensors

    alerts = alerts.sort_values("Anomaly_Score", ascending=False)

    st.dataframe(
        alerts[alert_columns].head(100).round(3),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# HISTORICAL RISK ANALYSIS
# --------------------------------------------------
st.header("6. Historical Risk Analysis")

if filtered["Timestamp"].notna().any():
    timeline = filtered.dropna(subset=["Timestamp"]).copy()
    timeline["Time Period"] = timeline["Timestamp"].dt.floor("h")

    timeline_summary = (
        timeline.groupby(["Time Period", "Risk_Level"])
        .size()
        .reset_index(name="Records")
    )

    fig_timeline = px.area(
        timeline_summary,
        x="Time Period",
        y="Records",
        color="Risk_Level",
        title="Risk Classification Over Time"
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.info(
        "A valid Date or Time column is required to display a chronological "
        "risk timeline."
    )

# --------------------------------------------------
# EFFICIENCY STATUS
# --------------------------------------------------
st.header("7. Efficiency Status Analysis")

if cols["efficiency"]:
    efficiency_data = filtered["Efficiency_Status"].value_counts().reset_index()
    efficiency_data.columns = ["Efficiency Status", "Records"]

    fig_efficiency = px.bar(
        efficiency_data,
        x="Efficiency Status",
        y="Records",
        color="Efficiency Status",
        title="Observed Manufacturing Efficiency Status"
    )
    st.plotly_chart(fig_efficiency, use_container_width=True)

    comparison = filtered.groupby("Efficiency_Status")[
        ["Anomaly_Score"] + usable_sensors
    ].mean().reset_index()

    st.dataframe(
        comparison.round(3),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Efficiency_Status is not available in this dataset.")

# --------------------------------------------------
# MACHINE PRIORITY SUMMARY
# --------------------------------------------------
st.header("8. Machine Inspection Priority")

priority = (
    filtered.groupby("Machine_ID")
    .agg(
        Average_Anomaly_Score=("Anomaly_Score", "mean"),
        Maximum_Anomaly_Score=("Anomaly_Score", "max"),
        High_Risk_Records=("Risk_Level", lambda x: (x == "High").sum()),
        Total_Records=("Risk_Level", "size")
    )
    .reset_index()
    .sort_values(
        ["High_Risk_Records", "Maximum_Anomaly_Score"],
        ascending=False
    )
)

st.dataframe(
    priority.round(2),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------
st.header("9. Export Results")

csv_output = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Anomaly Detection Results",
    data=csv_output,
    file_name="predictive_maintenance_results.csv",
    mime="text/csv"
)

priority_output = priority.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Machine Priority Report",
    data=priority_output,
    file_name="machine_inspection_priority.csv",
    mime="text/csv"
)

st.divider()
st.caption(
    "Unified Mentor Project | 6G-Integrated Smart Manufacturing Systems"
)
st.caption(
    "Anomaly scores and risk levels are model-based screening indicators, "
    "not confirmed failure probabilities. The dataset does not establish "
    "actual failure events or maintenance outcomes unless such labels exist."
)
