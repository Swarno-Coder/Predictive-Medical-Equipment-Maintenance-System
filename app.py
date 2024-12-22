import streamlit as st
import pandas as pd
import joblib  

@st.cache_data
def load_model():
    return [joblib.load('equipment_model.pkl'),
            joblib.load('equipment_model2.pkl'),
            joblib.load('equipment_model3.pkl'),
            joblib.load('equipment_model4.pkl')]  # Load your model file

model = load_model()

st.write("# Predictive Medical Equipment Maintenance System ⛑🔬")
st.write("### This system predicts the probability of equipment failure based on various factors.")
st.write("The model is a prototype trained on synthetic data and may not reflect real-world scenarios.")

st.write("**Please enter the equipment details to predict the failure probability.**")
st.write("---")

# Sidebar for User Inputs
st.sidebar.header("User Input Parameters")
equipment_type = st.sidebar.selectbox("Equipment Type", ['Blood Analyzer','CT Scanner', 'MRI Scanner', 'Ventilator', 'X-ray Machine'])
temperature = st.sidebar.slider("Temperature (°C)", 18.0, 40.1, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 20.0, 80.0, 45.0)
last_maintenance = st.sidebar.slider("Last Maintenance (days)", 0, 730, 180)
uptime = st.sidebar.slider("Uptime (hours)", 100, 5000, 1000)
sensor_1 = st.sidebar.slider("Sensor 1", 0.0, 1.0, 0.5)
sensor_2 = st.sidebar.slider("Sensor 2", 0.0, 1.0, 0.5)
vibration = st.sidebar.slider("Vibration (g)", 0.0, 6.0, 3.0)
voltage_fluctuation = st.sidebar.slider("Voltage Fluctuation (V)", 0.0, 1.5, 0.5)
pressure = st.sidebar.slider("Pressure (mmHg)", 80.0, 130.0, 100.0)

# Create input dataframe
input_data = pd.DataFrame({
    'equipment_type': [equipment_type],
    'temperature_avg': [temperature],
    'humidity_avg': [humidity],
    'last_maintenance_days': [last_maintenance],
    'uptime_hours': [uptime],
    'sensor_1': [sensor_1],
    'sensor_2': [sensor_2],
    'vibration': [vibration],
    'voltage_fluctuation': [voltage_fluctuation],
    'pressure': [pressure]
})

# Model Prediction
if st.sidebar.button('Predict'):
    
    prediction = [model[0].predict(input_data),
                  model[1].predict(input_data),
                  model[2].predict(input_data),
                  model[3].predict(input_data)]
    output = [pd.DataFrame(prediction[0], columns=['failure_probability']),
              pd.DataFrame(prediction[3], columns=['failure_reason']),
              pd.DataFrame(prediction[1], columns=['cost_implications']),
              pd.DataFrame(prediction[2], columns=['updated_uptime']),]#, 'failure_reason', 'cost_implications', 'updated_uptime'])
    failure_prob = output[0]['failure_probability'][0]

    if failure_prob < 0.3:
        health = 'Good'
        maintenance_level = 'Low'
        color = 'green'
        flag = 0 
    elif failure_prob < 0.6:
        health = 'Moderate'
        maintenance_level = 'Medium'
        color = '#DBA800'
        flag = 1
    else:
        health = 'Critical'
        maintenance_level = 'High'
        color = 'red'
        flag = 2
    
    st.write("### Prediction Results")
    st.metric(label="Predicted Failure Probability", value=f"{max(0, min(output[0]['failure_probability'][0] * 100, 100)):.1f}%")
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Equipment Health")
        st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;text-align:center;'>{health}</div>", unsafe_allow_html=True)
        # st_lottie(load_json(f"lottie{flag}.json"), key="loading")
    with col2:
        st.write("Maintenance Level Required")
        st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;text-align:center;'>{maintenance_level}</div>", unsafe_allow_html=True)
        # st_lottie(load_json(f"lottie{flag}.json"), key="loading")
    st.write("---")
    st.write(f"**Failure Reason:** {list(('Overheating', 'Component Wear', 'Voltage Surge', 'Pressure Leak', 'Sensor Drift'))[output[1]['failure_reason'][0]]}")
    st.write("---")
    st.write(f"**Cost Implications:** ${output[2]['cost_implications'][0]}")
    st.write("---")
    st.write(f"**Updated Uptime (hours):** {output[3]['updated_uptime'][0]}")
    
    print(input_data)
