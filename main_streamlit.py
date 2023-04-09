import streamlit as st
import numpy as np
import pandas as pd
import TimeSeriesCurveFitter as tscf

@st.cache_resource()
def read_data():
    data = pd.DataFrame(np.loadtxt('temperature.txt'))
    data = np.array(data).reshape(-1)
    return data

@st.cache_data(experimental_allow_widgets=True)
def optimize():
    period = st.slider('Period', 1, 365, 365)
    ar_order = st.slider('AR order(Trend component)', 1, 5, 1)
    smoothing_weight = st.slider('Smoothing weight', 1, 500, 100)

    data = read_data()

    model = tscf.TimeSeriesCurveFitter(data, period=period, ar_order=ar_order, smoothening_weight=smoothing_weight)
    final_estimate, residual, trend, cyclic, rms =  model.fit()
    
    return model, rms

st.title('Time Series Fitting using Convex Optimization')

model, rms = optimize()

# Streamlit RMS value
st.write(f'RMS: {np.round(rms, 2)}')

# Streamlit plot matplotlib pyplot
st.pyplot(model.generate_plots())