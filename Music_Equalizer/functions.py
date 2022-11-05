import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import streamlit_vertical_slider  as svs
from scipy.misc import electrocardiogram
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io.wavfile import write
import wave

#-------------------------------------- UNIFORM RANGE MODE ----------------------------------------------------
def uniform_range_mode(column1, column2, column3):

    column1.audio  (".piano_timpani_piccolo_out.wav", format='audio/wav') # displaying the audio
    obj = wave.open(".piano_timpani_piccolo_out.wav", 'rb')
    sample_rate   = obj.getframerate()      # number of samples per second
    n_samples     = obj.getnframes()        # total number of samples in the whole audio
    duration      = n_samples / sample_rate # duration of the audio file
    signal_wave   = obj.readframes(-1)      # amplitude of the sound
    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int16)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))

    yf = rfft(signal_y_axis)                                               # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform
    
    points_per_freq = len(xf) / (xf[-1])                                   # NOT UNDERSTANDABLE 

    # plotting_graphs(column2,signal_x_axis, signal_y_axis, False)

    columns=st.columns(10)
    index=0
    list_of_sliders_values = []
    while index < 10:
        with columns[index]:
            sliders = columns[index].slider(label="", key=index, min_value=0, max_value=10,value=1, step=1)
        index +=1
        list_of_sliders_values.append(sliders)
    
    for indexxx,value in enumerate(list_of_sliders_values):
        if value is not None:
            yf[int(points_per_freq * 1000 * indexxx)  : int(points_per_freq * 1000 * indexxx) + 1000] *= value
    else:
        pass

    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders 
    modified_signal_channel = np.int16(modified_signal) # returns two channels
    
    # plotting_graphs(column3,signal_x_axis, modified_signal, False)

    write(".Equalized_Music.wav", sample_rate, modified_signal_channel) # creates the modified song
    st.audio(".Equalized_Music.wav", format='audio/wav')

#-------------------------------------- MUSICAL INSTRUMENTS EQUALIZER ----------------------------------------------------
def musical_instruments_equalizer():

    st.audio('.piano_timpani_piccolo_out.wav', format='audio/wav')  # displaying the audio 
    obj         = wave.open(".piano_timpani_piccolo_out.wav", 'rb') # creating object 
    sample_rate = obj.getframerate()          # number of samples per second
    n_samples   = obj.getnframes()            # total number of samples in the whole audio
    duration    = n_samples / sample_rate     # duration of the audio file
    signal_wave = obj.readframes(-1)          # amplitude of the sound
    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int16)   #
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis)) #

    yf = rfft(signal_y_axis)                                               # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform
    
    points_per_freq = len(xf) / (xf[-1]) # NOT UNDERSTANDABLE 

    # fig, axs = plt.subplots()
    # fig.set_size_inches(14,5)
    # plt.plot(signal_x_axis,signal_y_axis)
    # st.plotly_chart(fig)

    slider_range_drum    = st.slider(label='Drum Sound'   , min_value=0, max_value=10, value=1, step=1, key="drum slider")
    slider_range_timpani = st.slider(label='Timpani Sound', min_value=0, max_value=10, value=1, step=1, key="timpani slider")
    slider_range_piccolo = st.slider(label='Piccolo Sound', min_value=0, max_value=10, value=1, step=1, key="piccolo slider")

    yf[int(points_per_freq*0)   :int(points_per_freq* 1000)] *= slider_range_drum
    yf[int(points_per_freq*1000):int(points_per_freq* 2600)] *= slider_range_timpani
    yf[int(points_per_freq*2700):int(points_per_freq*16000)] *= slider_range_piccolo

    # fig4, axs2 = plt.subplots()
    # fig4.set_size_inches(14,5)
    # plt.plot(xf,np.abs(yf))
    # st.plotly_chart(fig4,use_container_width=True)

    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders 
    modified_signal_channel = np.int16(modified_signal) # returns two channels 

    write   (".Equalized_Music.wav", sample_rate, modified_signal_channel) # creates the modified song
    st.audio(".Equalized_Music.wav", format='audio/wav')

#-------------------------------------- MEDICAL APPLICATION ----------------------------------------------------
def arrhythima(column1, column2, column3):

    ecg = electrocardiogram()       # Calling the arrhythmia database of a woman
    fs = 360                        # determining f sample
    time = np.arange(ecg.size) / fs # detrmining tima axis

    fourier_x_axis = rfftfreq(len(ecg), (time[1]-time[0])) # Computing fourier x axis
    fourier_y_axis = rfft(ecg)                             # Computing fourier y axis

    points_per_freq = len(fourier_x_axis) / (fourier_x_axis[-1]) # Duration

    fourier_y_axis[int(points_per_freq*1) : int(points_per_freq* 5)] *= 0 

    plotting_graphs(column2,time,ecg,True)

    modified_signal = irfft(fourier_y_axis) 

    plotting_graphs(column3, time, modified_signal, True)

#-------------------------------------- PLOTING ----------------------------------------------------
def plotting_graphs(column,x_axis,y_axis,flag):
    
    fig, axs = plt.subplots()
    fig.set_size_inches(6,3)
    plt.plot(x_axis,y_axis)
    if flag == True:
        plt.xlim(46, 50)
        plt.xlabel("Time in s")
        plt.ylabel("ECG in mV")
    column.plotly_chart(fig)
