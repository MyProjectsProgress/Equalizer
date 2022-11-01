import streamlit as st
import pandas as pd
import functions as fn
st.set_page_config(layout="wide")

# ------------------------------------------------------------------------------------ uploaded_file Browsing Button
uploaded_file = st.file_uploader(label="Uploading Signal", type = ['csv',"wav"])

# ------------------------------------------------------------------------------------ User Options
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Default Signal", "Music", "Vowels", "Arrhythima","Optional"])

# ------------------------------------------------------------------------------------Calling Main Functions
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    inverseFourier, fourierTransform = fn.fourier_transform(df)
    fn.fourier_inverse_transform(inverseFourier,df)
    fn.wave_ranges(fourierTransform)
else:
    st.write("HELLO MY FRIENDS, HOW ARE YOU TODAY?")

with tab1:
    names_list = [('A', 100),('B', 150),('C', 75),('D', 25),('E', 150),('F', 60),('G', 86),('H', 150),('E', 150),('G', 25),('K', 99),('L', 150),
                    ('M', 150),('M', 55),('N', 150)]
    fn.creating_sliders(names_list)

with tab2:
    names_list = [('Megzawy', 100),('Magdy', 150)]
    fn.creating_sliders(names_list)

with tab3:
    names_list = [('Amr', 100),('Sameh', 150),]
    fn.creating_sliders(names_list)

with tab4:
    names_list = [('Mariam', 100),('Taha', 150),]
    fn.creating_sliders(names_list)
