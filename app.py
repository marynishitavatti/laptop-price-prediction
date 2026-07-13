import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 style='text-align: center;'>💻 Laptop Price Prediction App</h1>", unsafe_allow_html=True)

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_excel("Laptop_price dataset.xlsx")
df = df.drop(columns=["Column1"], errors="ignore")

# -------------------------------
# Encode Brand
# -------------------------------
le = LabelEncoder()
df["Brand"] = le.fit_transform(df["Brand"])

# -------------------------------
# Features & Target
# -------------------------------
X = df[[
    "Brand",
    "Processor_Speed",
    "RAM_Size",
    "Storage_Capacity",
    "Screen_Size",
    "Weight"
]]
y = df["Price"]

# -------------------------------
# Train Models
# -------------------------------
lr = LinearRegression()
rf = RandomForestRegressor()
dt = DecisionTreeRegressor()

lr.fit(X, y)
rf.fit(X, y)
dt.fit(X, y)

# -------------------------------
# Model Performance
# -------------------------------
lr_acc = r2_score(y, lr.predict(X))
rf_acc = r2_score(y, rf.predict(X))
dt_acc = r2_score(y, dt.predict(X))

st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.write({
    "Linear Regression": f"{lr_acc*100:.2f}%",
    "Random Forest": f"{rf_acc*100:.2f}%",
    "Decision Tree": f"{dt_acc*100:.2f}%"
})

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter Laptop Details")

brand = st.sidebar.selectbox("Brand", le.classes_)
processor = st.sidebar.slider("Processor Speed", 1.0, 5.0, 2.5)
ram = st.sidebar.selectbox("RAM (GB)", [4, 8, 16, 32])
storage = st.sidebar.selectbox("Storage (GB)", [128, 256, 512, 1024])
screen = st.sidebar.slider("Screen Size", 10.0, 18.0, 13.0)
weight = st.sidebar.slider("Weight", 1.0, 3.0, 2.0)

# -------------------------------
# Model Selection Dropdown
# -------------------------------
st.markdown("### 🤖 Select Model")

model_choice = st.selectbox(
    "Choose a model for prediction",
    ["Random Forest", "Linear Regression", "Decision Tree"]
)

# -------------------------------
# Input Summary
# -------------------------------
st.subheader("Selected Configuration")
st.write({
    "Brand": brand,
    "Processor Speed": processor,
    "RAM": ram,
    "Storage": storage,
    "Screen Size": screen,
    "Weight": weight
})

st.markdown("---")

# -------------------------------
# Prepare Input
# -------------------------------
brand_encoded = le.transform([brand])[0]

input_df = pd.DataFrame({
    "Brand": [brand_encoded],
    "Processor_Speed": [processor],
    "RAM_Size": [ram],
    "Storage_Capacity": [storage],
    "Screen_Size": [screen],
    "Weight": [weight]
})

# -------------------------------
# Select Model Dynamically
# -------------------------------
if model_choice == "Random Forest":
    selected_model = rf
    model_name = "Random Forest Regressor"
elif model_choice == "Linear Regression":
    selected_model = lr
    model_name = "Linear Regression"
else:
    selected_model = dt
    model_name = "Decision Tree Regressor"

# -------------------------------
# Prediction
# -------------------------------
prediction = selected_model.predict(input_df)

# -------------------------------
# Model Used Display
# -------------------------------
st.markdown("### 🤖 Model Used")
st.info(model_name)

# -------------------------------
# Output
# -------------------------------
st.subheader("Predicted Price 💰")
st.success(f"₹ {prediction[0]:,.2f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built using Machine Learning & Streamlit 🚀")