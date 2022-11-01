import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

audio_file = st.file_uploader(label="", type=".wav")

if audio_file is not None:
    st.audio(audio_file, format='audio/ogg') #displaying the audio player
    signal, sample_freq = librosa.load(audio_file)  #getting audio attributes which are amplitude and frequency (number of frames per second)
    
    # Plotting audio Signal
    fig = plt.figure(figsize=[10,6])
    librosa.display.waveshow(signal, sample_freq)
    st.pyplot(fig)