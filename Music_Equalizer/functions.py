import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
import wave
import librosa
import librosa.display
import IPython.display as ipd
import os
import streamlit.components.v1 as components
from scipy import signal
import altair as alt
import pandas as pd
from scipy.io.wavfile import write


#-------------------------------------- MEDICAL APPLICATION ----------------------------------------------------
def arrhythmia(tools_col,graphs_col):
    ecg_dataset        = electrocardiogram()                                        # Calling the arrhythmia database of a woman
    sampling_frequency = 360                                                        # determining f sample
    time               = np.arange(ecg_dataset.size) / sampling_frequency           # detrmining time axis

    y_fourier, points_per_freq = fourier_transform(ecg_dataset, sampling_frequency) # Fourier Transfrom

    sliders_labels  = 'Arrhythmia'

    with tools_col:
        slider = vertical_slider()

    y_fourier[int(points_per_freq * 1) : int(points_per_freq * 5)] *= slider

    modified_signal = irfft(y_fourier) 

    static_graph(graphs_col, time, ecg_dataset, modified_signal)

#-------------------------------------- VOWELS ----------------------------------------------------


#-------------------------------------- VOICE TONE CHANGER ----------------------------------------------------
def voice_changer(uploaded_file, column1, column2):
    # voice = column1.radio('Voice', options=["Deep Voice", "Smooth Voice"])

    # if voice == "Deep Voice":
    #     empty = column2.empty()
    #     empty.empty()
    #     speed_rate           = 1.4
    #     sampling_rate_factor = 1.4

    # elif voice == "Smooth Voice":
    #     empty = column2.empty()
    #     empty.empty()
    #     speed_rate           = 0.5
    #     sampling_rate_factor = 0.5

    # loaded_sound_file, sampling_rate = librosa.load(uploaded_file, sr=None)
    # loaded_sound_file                = librosa.effects.time_stretch(loaded_sound_file, rate = speed_rate)

    # song = ipd.Audio(loaded_sound_file, rate = sampling_rate / sampling_rate_factor)
    # column2.write(type(song))
    # empty.write(song)


    voice = column1.radio('Voice', options=["Normal Voice","Deep Voice", "Smooth Voice"])
    
    if voice == "Normal Voice":
        Num_of_steps = 0

    if voice == "Deep Voice":
        Num_of_steps = -5

    elif voice == "Smooth Voice":
        Num_of_steps = 10

    signal, sample_rate = librosa.load(uploaded_file, sr=None)
    modified_signal =librosa.effects.pitch_shift(signal,sr=sample_rate,n_steps=Num_of_steps)
    write("voice_changed.wav", sample_rate, modified_signal)

    time =np.linspace(0,signal.shape[0]/sample_rate,signal.shape[0] )                           # read audio file
    start, pause, resume, space = st.columns([1.001,1.0,0.99,7])                                # Buttons Columns
    start_btn  = start.button(label='Start')
    pause_btn  = pause.button(label='P  ause')
    resume_btn = resume.button(label='Resume')

    with column2:
        Dynamic_graph(time,signal,modified_signal,start_btn,pause_btn,resume_btn,sample_rate,True)  # Plot Dynamic Graph
        st.audio("voice_changed.wav", format='audio/wav')

#-------------------------------------- CUSTOM SLIDER ----------------------------------------------------
parent_dir       = os.path.dirname(os.path.abspath(__file__))
build_dir        = os.path.join(parent_dir, "build")
_vertical_slider = components.declare_component("vertical_slider", path=build_dir)

def vertical_slider(key=None):                                      # The function to be called
    slider_value = _vertical_slider(key=key ,default=1)
    return slider_value



#-------------------------------------- READ AUDIO FILES ----------------------------------------------------
def read_audio(audio_file):
    obj = wave.open(audio_file, 'rb')                            # creating object
    sample_rate   = obj.getframerate()                           # number of samples per second
    n_samples     = obj.getnframes()                             # total number of samples in the whole audio
    signal_wave   = obj.readframes(-1)                           # amplitude of the sound
    duration      = n_samples / sample_rate                      # duration of the audio file
    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int16)   # get the signal from buffer memory
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))

    return signal_x_axis, signal_y_axis, sample_rate
    
#-------------------------------------- FOURIER TRANSFORM ----------------------------------------------------
def fourier_transform(signal_y_axis, sample_rate):
    
    y_fourier       = rfft(signal_y_axis)                                # returns complex numbers of the y axis in the data frame
    x_fourier       = rfftfreq(len(signal_y_axis), 1/sample_rate)        # returns the frequency x axis after fourier transform
    points_per_freq = len(x_fourier) / (x_fourier[-1])                   # how many points in the y_fourier array from freq 0 to 1 
    return y_fourier, points_per_freq

#-------------------------------------- CREATE SLIDERS & MODIFY SIGNALS ----------------------------------------------------
def f_ranges(y_fourier, points_per_freq, n_sliders, sliders_labels, mode):

    columns = st.columns(n_sliders)                                     # Create sliders and its' labels
    counter = 0
    list_of_sliders_values = []
    while counter < n_sliders:
        with columns[counter]:
            st.write(sliders_labels[counter])
            sliders = (vertical_slider(counter))
        counter += 1
        list_of_sliders_values.append(sliders)

    if   mode == "Default":                                            # Attache certain frequencies to each slider depending on the choosen mode
        for index,value in enumerate(list_of_sliders_values):
            y_fourier[int(points_per_freq * 1000 * index)  : int(points_per_freq * 1000 * index + points_per_freq * 1000)] *= value

    elif mode == "Music"  :
        y_fourier[                         :int(points_per_freq* 1000)] *= list_of_sliders_values[0] * .35
        y_fourier[int(points_per_freq*1000):int(points_per_freq* 2600)] *= list_of_sliders_values[1]
        y_fourier[int(points_per_freq*2600):                          ] *= list_of_sliders_values[2] * 0.6

    elif mode == "Vowels" :
        # sliders_labels = ['ʃ','ʊ','a','r','b']

        # /ʃ/ range
        y_fourier[int(800*points_per_freq):int(5000*points_per_freq)]  *= list_of_sliders_values[0]
       
        #/ʊ/ range
        y_fourier[int(500*points_per_freq):int(2000*points_per_freq)]  *= list_of_sliders_values[1]

        # /a/ range
        y_fourier[int(500*points_per_freq):int(1200*points_per_freq)]  *= list_of_sliders_values[2]
        ##500-2000 try
        
        # /r/ range
        y_fourier[int(900*points_per_freq):int(5000*points_per_freq)]  *= list_of_sliders_values[3]
        # y_fourier *=2

        # /b/ range
        y_fourier[int(1200*points_per_freq):int(5000*points_per_freq)]  *= list_of_sliders_values[4]
        # y_fourier *=2



    elif mode == "Arrhythmia":
        # y_fourier[int(points_per_freq * 1) : int(points_per_freq * 5)] *= list_of_sliders_values[0]
        pass 

    elif mode == "Voice Tone Changer":
        pass

    return y_fourier

#--------------------------------------  STATIC PLOTTING ----------------------------------------------------
def static_graph(column, x_axis, y_axis1, y_axis2 = None):

    if y_axis2 is not None:

        fig= plt.figure(figsize=[15,5])
        plt.plot   (x_axis, y_axis2)
        plt.xlim   (45, 51)
        plt.title  ("Modified ecg_dataset")
        plt.xlabel ("Time in s")
        plt.ylabel ("Amplitude in mV")
        plt.grid()

    else:
        fig= plt.figure(figsize=[15,5])
        plt.plot  (x_axis,y_axis1)
        plt.title ("Audio")
        plt.xlabel("Time in s")
        plt.ylabel("Amplitde")
        plt.grid()

    column.pyplot(fig)

#-------------------------------------- PLOTTING SPECTROGRAM ----------------------------------------------------
def plot_spectro(original_audio, modified_audio):
    
    y1, sr = librosa.load(original_audio)
    y2, sr = librosa.load(modified_audio)
    D1     = librosa.stft(y1)             # STFT of y
    S_db1  = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
    D2     = librosa.stft(y2)             # STFT of y
    S_db2  = librosa.amplitude_to_db(np.abs(D2), ref=np.max)

    fig= plt.figure(figsize=[15,8])
    plt.subplot(2,2,1)
    img1 = librosa.display.specshow(S_db1, x_axis='time', y_axis='linear')
    plt.subplot(2,2,2)
    img2 = librosa.display.specshow(S_db2, x_axis='time', y_axis='linear')
    # fig.colorbar(img1, format="%+2.f dB")
    # fig.colorbar(img2, format="%+2.f dB")

    st.pyplot(fig)

#-------------------------------------- DYNAMIC PLOTTING ----------------------------------------------------
class Variables:
    start=0

def plot_animation(df,flag):
    brush  = alt.selection_interval ()

    if flag:
        chart1 = alt.Chart(df).mark_line().encode(x=alt.X('time', axis=alt.Axis(title='Time')),).properties(width=414,height=300).add_selection(brush).interactive()
    else:
        chart1 = alt.Chart(df).mark_line().encode(x=alt.X('time', axis=alt.Axis(title='Time')),).properties(width=414,height=200).add_selection(brush).interactive()
    
    figure = chart1.encode(y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude'))) | chart1.encode(y ='amplitude after processing').add_selection(brush)
    return figure

def Dynamic_graph(signal_x_axis, signal_y_axis, signal_y_axis1,start_btn,pause_btn,resume_btn,sample_rate,flag):

        step_plot= int(sample_rate/210)

        df = pd.DataFrame({'time': signal_x_axis[::step_plot], 'amplitude': signal_y_axis[:: step_plot], 'amplitude after processing': signal_y_axis1[::step_plot]}, columns=['time', 'amplitude','amplitude after processing'])

        lines       = plot_animation(df,flag)
        line_plot   = st.altair_chart(lines)

        df_elements = df.shape[0] # number of elements in the dataframe
        burst       = 10          # number of elements  to add to the plot
        size        = burst       # size of the current dataset

        if start_btn:
            for i in range(1, df_elements):
                Variables.start      = i
                step_df              = df.iloc[0:size]
                lines                = plot_animation(step_df,flag)
                line_plot            = line_plot.altair_chart(lines)
                Variables.graph_size = size
                size                 = i * burst 

        if resume_btn: 
            for i in range(Variables.start,df_elements):
                Variables.start      = i
                step_df              = df.iloc[0:size]
                lines                = plot_animation(step_df,flag)
                line_plot            = line_plot.altair_chart(lines)
                Variables.graph_size = size
                size                 = i * burst

        if pause_btn:
            step_df   = df.iloc[0:Variables.graph_size]
            lines     = plot_animation(step_df,flag)
            line_plot = line_plot.altair_chart(lines)
