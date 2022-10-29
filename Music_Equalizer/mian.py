import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd

# ------------------------------------------------------------------------------------ User Options
dataset = st.sidebar.file_uploader(label="Uploading Signal", type = ['csv'])

# ------------------------------------------------------------------------------------Calling Main Functions
if dataset is not None:
    df = pd.read_csv(dataset)

else:
    pass