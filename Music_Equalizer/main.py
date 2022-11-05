import streamlit as st
import pandas as pd
import functions as fn

st.set_page_config(layout="wide")
with open('Music_Equalizer\main.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)

#------------------------------------------------------------------------------------- COLUMNS

column1,column2,column3=st.columns([1,3,3])

# ------------------------------------------------------------------------------------ Uploaded File Browsing Button
uploaded_file = column1.file_uploader(label="Uploading Signal", type = ['csv',".wav"])

#------------------------------------------------------------------------------------- USER OPTIONS

select_mode = column1.selectbox("",[ "Default","Music", "Vowels", "Arrhythima", "Optional"])

# ------------------------------------------------------------------------------------ Calling Main Functions
if uploaded_file is not None:

    # Determining whether the file is csv or wav
    file_name = uploaded_file.type
    file_extension = file_name[-3:]

    if select_mode =="Default":
        fn.uniform_range_mode(column1, column2, column3, uploaded_file)

    elif select_mode == "Music":
        if file_extension == "wav":
            fn.musical_instruments_equalizer(uploaded_file)

    elif select_mode == "Vowels":
        pass

    elif select_mode == "Arrhythima":
            fn.arrhythima()

    else:
        pass

else:
    pass
    # fn.uniform_range_mode(column1, column2, column3)
