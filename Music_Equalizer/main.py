import streamlit as st
import pandas as pd
import functions as fn

st.set_page_config(layout="wide")

st.markdown("""
            <h1 style='text-align: center; margin-bottom: 30px; margin-top:-25px'>
            Equalizer Studio
            </h1>""", unsafe_allow_html=True
            )

with open('main.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)

#------------------------------------------------------------------------------------- COLUMNS
column1,column2,column3=st.columns([1,3,3])

# ------------------------------------------------------------------------------------ Uploaded File Browsing Button
uploaded_file = column1.file_uploader(label="Uploading Signal", type = ['csv',".wav"])

#------------------------------------------------------------------------------------- USER OPTIONS
select_mode = column1.selectbox("",[ "Default","Music", "Vowels", "Arrhythima", "Optional"])

# ------------------------------------------------------------------------------------ Show Spectrogram box
show_spectro = column1.checkbox("Show Spectrogram")

# ------------------------------------------------------------------------------------ Calling Main Functions

if uploaded_file is not None:
    file_name = uploaded_file.type
    file_extension = file_name[-3:]

if select_mode == "Default":
    fn.uniform_range_mode(column1, column2, column3,show_spectro)

elif select_mode == "Music":
    fn.musical_instruments_equalizer(column1, column2, column3, show_spectro)

elif select_mode == "Vowels":
    fn.vowels_equalizer(column1, column2, column3, show_spectro)

elif select_mode == "Arrhythima":
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        fn.arrhythima(column1, column2, column3 ,show_spectro, df)

elif select_mode == "Optional":
    if uploaded_file:
        fn.voice_changer(uploaded_file, column1, column2, column3, show_spectro)