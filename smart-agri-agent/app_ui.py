import streamlit as st
from app import run_agent

st.set_page_config(page_title="Smart Agriculture Advisor")

st.title("🌾 Smart Agriculture Advisor Agent")

st.write("Enter soil and weather details:")

n = st.number_input("Nitrogen (N)", min_value=0.0)
p = st.number_input("Phosphorus (P)", min_value=0.0)
k = st.number_input("Potassium (K)", min_value=0.0)
temp = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

if st.button("🌱 Get Recommendation"):
    input_data = f"{n},{p},{k},{temp},{humidity},{rainfall}"
    
    with st.spinner("Thinking... 🤖"):
        result = run_agent(input_data)
    
    st.success(result)