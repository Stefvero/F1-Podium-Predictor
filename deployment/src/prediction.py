import streamlit as st
from joblib import load
import pickle as pkl
import json
import pandas as pd
import numpy as np

model = load('./src/best_model.pkl')

with open('./src/nationalities.txt', 'rb') as file_2:
  nats = json.load(file_2)

driver_nats = nats["driver_nationalities"]
constructor_nats = nats["constructor_nationalities"]


def run():
    with st.form(key = 'form_ftds_rmt_053'):
        grid_position = st.slider("Grid Position", 1, 26, 5, help= "Driver's Grid Position at Race Day")
        quali_position = st.slider("Qualifying Position", 1, 26, 5, help= "Driver's Qualifying Position before Race Day")
        round_num = st.slider("Round of Season", 1, 26, 5, help= "Which Race of the Season")
        year = st.number_input("Year", min_value = 1950, max_value = 2026, value= 2020, help="Season Year (1950-2026)")
        driver_position_before = st.slider("Driver's Championship Position", 1, 26, 5, help= "Driver's Position on the World Driver's Championship")
        driver_points_before = st.number_input("Points Before Race", min_value = 0, max_value = 600, value= 100, help="Driver Points in the Championship (0-600)")
        driver_wins_before = st.slider("Driver's Total Wins", 1, 110, 5, help= "Driver's Total Wins in their career")
        driver_age = st.number_input("Age", min_value = 15.0, max_value = 100.0, value = 20.0, step = 0.5, help = 'Driver Age')
        driver_nationality = st.selectbox("Driver Nationality", options=driver_nats, index=driver_nats.index("Dutch") if "Dutch" in driver_nats else 0,help="Driver's nationality. Must match a nationality seen during model training.")
        constructor_nationality = st.selectbox("Constructor Nationality", options=constructor_nats, index=constructor_nats.index("British") if "British" in constructor_nats else 0, help="The nationality of the constructor (team). Must match a nationality seen during model training.")
        constructor_position_before = st.slider("Constructor's Championship Position", 1, 13, 5, help= "Constructor's Position on the World Constructor's Championship")
        driver_historical_dnf_rate = st.slider("DNF Rate", 0.0, 1.0, 0.01, help= "Driver's Historical DNF Rate")
        pit_stops = st.slider("Pit Stops", 0, 10, help= "Pit Stops done within the race")

        submitted = st.form_submit_button('Predict')


    input_df = pd.DataFrame([{
       "grid_position" : float(grid_position),
       "quali_position" : float(quali_position),
       "round" : int(round_num),
       "year" : int(year),
       "driver_points_before" : float(driver_points_before),
       "driver_position_before" : float(driver_position_before),
       "driver_wins_before" : float(driver_wins_before),
       "driver_age" : float(driver_age),
       "driver_nationality" : driver_nationality,
       "constructor_nationality" : constructor_nationality,
       "constructor_position_before" : float(constructor_position_before),
       "driver_historical_dnf_rate" : float(driver_historical_dnf_rate),
       "pit_stops" : float(pit_stops),
        }])

    if submitted:
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        podium_prob = probability[1]
        no_podium_prob = probability[0]
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("This driver is likely to finish on the podium")
        else:
            st.warning("This driver is unlikely to finish on the podium.")
        
        st.write("### Probabilities")
        st.write(f"Podium Probability: {podium_prob:.2%}")
        st.write(f"No Podium Probability: {no_podium_prob:.2%}")
        


if __name__ == '__main__':
  run()