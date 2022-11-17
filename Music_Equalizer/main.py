import streamlit as st
import functions as fn
from scipy.fft import irfft
from scipy.io.wavfile import write
import numpy as np

#-------------------------------------------------------------------- SETTING PAGE LAYOUT -----------------------------------------------------
st.set_page_config(layout="wide")

with open('main.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)

tools_col, space, graphs_col=st.columns([1.8,0.2,6])

#-------------------------------------------------------------------- FILE UPLOADER & MODE --------------------------------------------------
with tools_col:
    st.markdown("""
            <h3 style='text-align: center; margin-bottom: 0px; margin-top:-15px'>
            Equalizer Studio
            </h3>""", unsafe_allow_html=True
            )

uploaded_file = tools_col.file_uploader(label="", type = ['csv',".wav"])

select_mode   = tools_col.selectbox("Mode", ["Default","Music", "Vowels", "Arrhythmia", "Voice Tone Changer"])

# ------------------------------------------------------------------- GETTING FILE NAME --------------------------------------------------
if uploaded_file is not None:
    file_name = uploaded_file.name

#-------------------------------------------------------------------- CHOOSING MODE -----------------------------------------------------------
if select_mode == "Default":
    uploaded_file = "BabyElephantWalk.wav"
    file_name     = "BabyElephantWalk.wav"
    number_of_sliders = 10
    ranges = [[0,1000],[1000,2000],[2000,3000],[3000,4000],[4000,5000],[5000,6000],[6000,7000],[7000,8000],[8000,9000],[9000,10000]]
    sliders_labels = ['0 to 1k Hz', '1k to 2k Hz', '2k to 3k Hz','3k to 4k Hz',
    '4k to 5k Hz', '5k to 6k Hz','6k to 7k Hz', '7k to 8k Hz', '8k to 9k Hz','9k to 10k Hz']

elif select_mode == "Music":
    number_of_sliders = 3
    ranges = [[0,1000],[1000,2600],[2600,22049]]
    sliders_labels = ['Drums', 'Timpani', 'Piccolo']

elif select_mode == "Vowels":
    number_of_sliders = 5
    ranges = [[800,5000],[500,2000],[500,1200],[900,5000],[1200,5000]]
    sliders_labels = ['ʃ','ʊ','a','r','b']

elif select_mode == "Arrhythmia":
        fn.arrhythmia(tools_col,graphs_col)

elif select_mode == "Voice Tone Changer":
    if uploaded_file:
        fn.voice_changer(uploaded_file, tools_col, graphs_col)

#--------------------------------------------------------------------- MAIN FUNCTION ---------------------------------------------------------

if uploaded_file is not None and select_mode != "Arrhythmia" and select_mode != "Voice Tone Changer":

    show_spectro  = tools_col.checkbox("Show Spectrogram")                                              # Spectrogram Checkbox

    start, pause, resume, space = st.columns([1.001,1.01,0.99,7])                                       # Buttons Columns

    signal_x_axis, signal_y_axis, sample_rate = fn.read_audio(uploaded_file)                            # read audio file

    y_fourier, points_per_freq = fn.fourier_transform(signal_y_axis, sample_rate)                       # Fourier Transfrom

    y_fourier = fn.f_ranges(y_fourier, points_per_freq, number_of_sliders, sliders_labels,ranges, select_mode) # create sliders and modify signal

    modified_signal         = irfft(y_fourier)                            # returns the inverse transform after modifying it with sliders
    modified_signal_to_audio = np.int16(modified_signal)                  # return audio to one channel 

    write(".Equalized_audio.wav", sample_rate, modified_signal_to_audio)  # creates the modified song

    with graphs_col:
        if (show_spectro):
            fn.plot_spectro(file_name,".Equalized_audio.wav")
        else:
            start_btn  = start.button(label='Start')
            pause_btn  = pause.button(label='Pause')
            resume_btn = resume.button(label='Resume')

            fn.Dynamic_graph(signal_x_axis,signal_y_axis,modified_signal,start_btn,pause_btn,resume_btn,sample_rate,False)  # Plot Dynamic Graph

    graphs_col.audio(".Equalized_audio.wav", format='audio/wav')                                             # displaying the audio after  editing
