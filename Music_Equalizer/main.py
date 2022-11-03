import streamlit as st
import pandas as pd
import functions as fn

st.set_page_config(layout="wide")

# ------------------------------------------------------------------------------------ Uploaded File Browsing Button
uploaded_file = st.file_uploader(label="Uploading Signal", type = ['csv',".wav"])

# ------------------------------------------------------------------------------------ Calling Main Functions
if uploaded_file is not None:

    # USER OPTIONS
    radio_button = st.radio("",["Default Signal", "Music", "Vowels", "Arrhythima", "Optional"])

    if radio_button == "Default Signal":
        pass

    elif radio_button == "Music":
        pass

    elif radio_button == "Vowels":
        pass

    elif radio_button == "Arrhythima":
        pass

    else:
        pass

    file_name = uploaded_file.type
    file_extension = file_name[-3:]

    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
        fn.dataframe_fourier_transform(df)
    else:
        fn.audio_fourier_transform(uploaded_file)

else:
    pass


