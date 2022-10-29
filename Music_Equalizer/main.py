import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import functions as fn

# ------------------------------------------------------------------------------------ User Options
dataset = st.sidebar.file_uploader(label="Uploading Signal", type = ['csv'])

# ------------------------------------------------------------------------------------Calling Main Functions
if dataset is not None:
    df = pd.read_csv(dataset)
    inverseFourier, fourierTransform = fn.fourier_transform(df)
    fn.fourier_inverse_transform(inverseFourier,df)
    fn.wave_ranges(fourierTransform)

else:
    pass