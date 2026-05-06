import streamlit as st
import eda
import prediction

st.set_page_config(
    page_title= 'F1 Podiums',
    layout='wide',
    initial_sidebar_state='expanded'
)

page = st.sidebar.selectbox('Choose Page', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()
else:
    prediction.run()