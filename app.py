import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from processor import DataProcessor
from optimizer import GridOptimizer

# Page Config
st.set_page_config(page_title="Smart Grid AI", layout="wide")
st.title("⚡ Smart Grid Load Balancing System")

# Sidebar Controls
st.sidebar.header("Grid Settings")
threshold_percent = st.sidebar.slider("Grid Load Threshold (%)", 0.0, 1.0, 0.5)
battery_size = st.sidebar.slider("Battery Capacity (Units)", 1.0, 10.0, 5.0)

# 1. Load Data & Model
@st.cache_resource
def load_resources():
    proc = DataProcessor('data/PJME_hourly.csv')
    data, scaler = proc.load_and_process()
    # Load Scikit-Learn model
    model = joblib.load('grid_ai_model.pkl')
    return proc, data, scaler, model

processor, raw_data, scaler, model = load_resources()

# 2. Run Inference (Predict Next 24 Hours)
st.subheader("Simulating Next 24 Hours...")
slice_start = 18000 
# Get the initial 24 hours of data
current_sequence = raw_data[slice_start:slice_start+24].flatten()

predictions = []
# Simulation Loop
for i in range(24):
    # Reshape to (1, 24) because model expects a 2D array
    input_data = current_sequence.reshape(1, -1)
    
    pred = model.predict(input_data)[0]
    predictions.append(pred)
    
    # Update sequence: Remove first hour, add new prediction at the end
    current_sequence = np.append(current_sequence[1:], pred)

# Reshape for inverse transform
predictions = np.array(predictions).reshape(-1, 1)
predictions_mw = scaler.inverse_transform(predictions)
threshold_mw = threshold_percent * (predictions_mw.max()) 

# 3. Optimize
optimizer = GridOptimizer(threshold=threshold_mw, battery_max=battery_size * 1000)
optimized_load, battery_history = optimizer.optimize(predictions_mw)

# 4. Visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(predictions_mw, label='Unbalanced Load (AI Prediction)', color='red', linestyle='--')
ax.plot(optimized_load, label='Balanced Load (After AI Adjustment)', color='green', linewidth=2)
ax.axhline(y=threshold_mw, color='blue', linestyle=':', label='Grid Threshold')
ax.set_title("Grid Load Balancing: Before vs After")
ax.set_ylabel("Power Consumption (MW)")
ax.set_xlabel("Hours Forecast")
ax.legend()
st.pyplot(fig)

# 5. Metrics & Status Indicator
st.write("### Impact Analysis")
col1, col2, col3 = st.columns(3)

peak_reduction = np.max(predictions_mw) - np.max(optimized_load)
col1.metric("Peak Load Reduced By", f"{int(peak_reduction)} MW")
col2.metric("Battery Remaining", f"{int(battery_history[-1])} MW")

# Status Logic: Check if the Green line ever went above the threshold
max_optimized_load = np.max(optimized_load)
if max_optimized_load > threshold_mw:
    # If even after balancing, we are still too high -> Error
    diff = int(max_optimized_load - threshold_mw)
    col3.error(f"⚠️ GRID UNSTABLE (Overloaded by {diff} MW)")
else:
    # If we stayed under the limit -> Success
    col3.success("✅ GRID STABLE")