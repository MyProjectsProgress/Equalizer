import streamlit as st
import pandas as pd
import functions as fn


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


st.set_page_config(layout="wide")

with open('main.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)

#------------------------------------------------------------------------------------- COLUMNS
column1, space, column2, column3=st.columns([1.8,0.2,3.01,3])

with column1:
    st.markdown("""
            <h3 style='text-align: center; margin-bottom: 0px; margin-top:0px'>
            Equalizer Studio
            </h3>""", unsafe_allow_html=True
            )

# ------------------------------------------------------------------------------------ Uploaded File Browsing Button
uploaded_file = column1.file_uploader(label="Uploading Signal", type = ['csv',".wav"])

#------------------------------------------------------------------------------------- USER OPTIONS
select_mode = column1.selectbox("Mode",[ "Default","Music", "Vowels", "Arrhythima", "Optional", "Animation"])

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


elif select_mode == "Animation":
    df = pd.read_csv(uploaded_file)
    time_data = df[df.head(0).columns[0]]
    amplitude_data = df[df.head(0).columns[1]]

    fig2 = plt.figure(figsize=[10,6])
    plt.plot(time_data,amplitude_data)
    column2.pyplot(fig2)