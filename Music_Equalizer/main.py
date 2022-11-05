import streamlit as st
import pandas as pd
import functions as fn
import wave

st.set_page_config(layout="wide")

#------------------------------------------------------------------------------------- COLUMNS
column1,column2,column3=st.columns([1,3,3])

# ------------------------------------------------------------------------------------ Uploaded File Browsing Button
uploaded_file = column1.file_uploader(label="Uploading Signal", type = ['csv',".wav"])

#------------------------------------------------------------------------------------- USER OPTIONS
radio_button = column1.radio("",[ "Default","Music", "Vowels", "Arrhythima", "Optional"])

# ------------------------------------------------------------------------------------ Calling Main Functions
if uploaded_file is not None:

    # Determining whether the file is csv or wav
    file_name = uploaded_file.type
    file_extension = file_name[-3:]

    if radio_button =="Default":
        pass

    elif radio_button == "Music":
        if file_extension == "wav":
            fn.musical_instruments_equalizer(uploaded_file)

    elif radio_button == "Vowels":
        pass

    elif radio_button == "Arrhythima":
            fn.arrhythima()

    else:
        pass

else:
    fn.uniform_range_mode(column1, column2, column3)
