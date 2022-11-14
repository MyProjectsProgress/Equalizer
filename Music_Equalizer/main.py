import streamlit as st
import functions as fn
import pandas as pd
from scipy.fft import irfft
from scipy.io.wavfile import write
import numpy as np

#-------------------------------------------------------------------- SETTING PAGE LAYOUT -----------------------------------------------------
st.set_page_config(layout="wide")

with open('main.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)

column1, space, column2=st.columns([1.8,0.2,6])

#-------------------------------------------------------------------- FILE UPLOADER & MODE --------------------------------------------------
with column1:
    st.markdown("""
            <h3 style='text-align: center; margin-bottom: 0px; margin-top:-15px'>
            Equalizer Studio
            </h3>""", unsafe_allow_html=True
            )

uploaded_file = column1.file_uploader(label="", type = ['csv',".wav"])

select_mode = column1.selectbox("Mode",[ "Default","Music", "Vowels", "Arrhythima", "Optional"])

show_spectro = column1.checkbox("Show Spectrogram")

# ------------------------------------------------------------------- GETTING FILE EXTENSION --------------------------------------------------
if uploaded_file is not None:
    file_type = uploaded_file.type
    file_extension = file_type[-3:]
    file_name = uploaded_file.name

#-------------------------------------------------------------------- CHOOSING MODE -----------------------------------------------------------
if select_mode == "Default":
    uploaded_file = ".piano_timpani_piccolo_out.wav"
    file_name = ".piano_timpani_piccolo_out.wav"
    n = 10
    sliders_labels = ['0 to 1k Hz', '1k to 2k Hz', '2k to 3k Hz','3k to 4k Hz',
    '4k to 5k Hz', '5k to 6k Hz','6k to 7k Hz', '7k to 8k Hz', '8k to 9k Hz','9k to 10k Hz']

elif select_mode == "Music":
    n = 3
    sliders_labels = ['Drums', 'Timpani', 'Piccolo']

elif select_mode == "Vowels":
    n = 5
    sliders_labels = ['Z','/i:/','/e/','ʊə','F']


elif select_mode == "Arrhythima":

 fn.arrhythima(column2, column1 ,show_spectro)

elif select_mode == "Optional":
    if uploaded_file:
        fn.voice_changer(uploaded_file, column1, column2, show_spectro)

#--------------------------------------------------------------------- MAIN FUNCTION ---------------------------------------------------------

if uploaded_file is not None and select_mode != "Arrhythima" and select_mode != "Optional":

    signal_x_axis, signal_y_axis, sample_rate = fn.read_audio(uploaded_file)       # read audio file

    y_fourier, points_per_freq = fn.fourier_transform(signal_y_axis, sample_rate)         # Fourier Transfrom

    y_fourier = fn.f_ranges(y_fourier, points_per_freq, n, sliders_labels, select_mode)         #create sliders and modify signal

    modified_signal         = irfft(y_fourier)                                            # returns the inverse transform after modifying it with sliders
    modified_signal_channel = np.int16(modified_signal)                            # returns two channels 

    write(".Equalized_audio.wav", sample_rate, modified_signal_channel)            # creates the modified song

    with column2:
        if (show_spectro):
            fn.plot_spectro(file_name,".Equalized_audio.wav")
        else:
            start_btn  = column1.button(label='Start')
            pause_btn  = column1.button(label='Pause')
            resume_btn = column1.button(label='resume')
            
            fn.Dynamic_graph(signal_x_axis,signal_y_axis,modified_signal,start_btn,pause_btn,resume_btn)


    # column2.audio(uploaded_file, format='audio/wav')                             # displaying the audio before editing
    column2.audio(".Equalized_audio.wav", format='audio/wav')                      # displaying the audio after  editing