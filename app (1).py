
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('army_heat_stress_model.pkl')
columns = joblib.load('feature_columns.pkl')

st.set_page_config(
    page_title="Army Heat Stress Prediction",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Army Soldier Heat Stress Risk Prediction")
st.write("Machine Learning Based Educational Prototype")

st.divider()

st.subheader("Enter Soldier Details")

col1, col2 = st.columns(2)

with col1:
    temp = st.slider("Temperature (°C)", 25.0, 45.0, 35.0)
    humidity = st.slider("Humidity (%)", 20, 95, 60)
    heart_rate = st.slider("Heart Rate", 60, 180, 100)
    body_temp = st.slider("Body Temperature (°C)", 36.0, 40.0, 37.0)

with col2:
    activity = st.selectbox("Activity Level", ["Low", "Medium", "High"])
    duration = st.slider("Exercise Duration (Minutes)", 10, 180, 60)
    hydration = st.selectbox("Hydration", ["Adequate", "Limited"])

if st.button("Predict Heat Stress Risk"):

    soldier = pd.DataFrame([{
        'Temperature': temp,
        'Humidity': humidity,
        'Heart_Rate': heart_rate,
        'Body_Temperature': body_temp,
        'Activity_Level': activity,
        'Exercise_Duration': duration,
        'Hydration': hydration
    }])

    soldier = pd.get_dummies(
        soldier,
        columns=['Activity_Level', 'Hydration']
    )

    soldier = soldier.reindex(columns=columns, fill_value=0)

    prediction = model.predict(soldier)[0]

    st.subheader("Prediction Result")

    st.success(f"Predicted Risk Level: {prediction}")

    st.warning(
        "Educational prototype using synthetic data. "
        "Not validated for medical or real-world Army safety decisions."
    )
