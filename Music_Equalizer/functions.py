import random 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit_vertical_slider  as svs
from scipy.fftpack import fft
from scipy.fft import fft, fftfreq, fftshift
from scipy.misc import electrocardiogram
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io.wavfile import write
from scipy.signal import find_peaks
import wave

#  ----------------------------------- TRIAL FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def dataframe_fourier_transform(dataframe):

    signal_x_axis = (dataframe.iloc[:,0]).to_numpy() # dataframe x axis
    signal_y_axis = (dataframe.iloc[:,1]).to_numpy() # dataframe y axis

    duration    = signal_x_axis[-1] # the last point in the x axis (number of seconds in the data frame)
    sample_rate = len(signal_y_axis)/duration # returns number points per second

    fourier_x_axis = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform
    fourier_y_axis = rfft(signal_y_axis) # returns complex numbers of the y axis in the data frame
    peaks = find_peaks(signal_y_axis) # computes peaks of the signal 
    peaks_indeces = peaks[0]  # list of indeces of frequency with high peaks

    points_per_freq = len(fourier_x_axis) / (sample_rate) # NOT UNDERSTANDABLE 
    
    fourier_y_axis = dataframe_creating_sliders(peaks_indeces, points_per_freq, fourier_x_axis, fourier_y_axis) # calling creating sliders function

    dataframe_fourier_inverse_transform(fourier_y_axis,signal_x_axis)

    # write("filename.wav", 44100, signal_y_axis)

    fig, axs = plt.subplots()
    fig.set_size_inches(14,5)
    plt.plot(fourier_x_axis, np.abs(fourier_y_axis)) #plotting signal before modifying
    plt.plot(fourier_x_axis[peaks_indeces[:]], np.abs(fourier_y_axis)[peaks_indeces[:]], marker="o") # plotting peaks points
    st.plotly_chart(fig,use_container_width=True)

#  ----------------------------------- DATAFRAME INVERSE FOURIER TRANSFORM ---------------------------------------------------
def dataframe_fourier_inverse_transform(fourier_y_axis,signal_x_axis):

    modified_signal = irfft(fourier_y_axis) # returning the inverse transform after modifying it with sliders
    fig2, axs2 = plt.subplots()
    fig2.set_size_inches(14,5)
    plt.plot(signal_x_axis,modified_signal) # ploting signal after modifying
    st.plotly_chart(fig2,use_container_width=True)

#  ----------------------------------- CREATING SLIDERS ---------------------------------------------------------------
def dataframe_creating_sliders(peaks_indeces,points_per_freq,fourier_x_axis,fourier_y_axis):

    peak_frequencies = fourier_x_axis[peaks_indeces[:]] 
    columns = st.columns(10)
    for index, frequency in enumerate(peak_frequencies): 
        with columns[index]:
            slider_range = svs.vertical_slider(min_value=0.0, max_value=2.0, default_value=1.0, step=.1, key=index)
        if slider_range is not None:
            fourier_y_axis[peaks_indeces[index]  - 2 : peaks_indeces[index]  + 2] *= slider_range
    return fourier_y_axis

#-------------------------------------- Musical Instruments Equalizer ----------------------------------------------------
def musical_instruments_equalizer(audio_file):

    st.audio(audio_file, format='audio/wav')  # displaying the audio
    obj         = wave.open(audio_file, 'rb') # creating object 
    sample_rate = obj.getframerate()          # number of samples per second
    n_samples   = obj.getnframes()            # total number of samples in the whole audio
    duration    = n_samples / sample_rate     # duration of the audio file
    signal_wave = obj.readframes(-1)          # amplitude of the sound

    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int16)   #
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis)) #

    yf = rfft(signal_y_axis)                                               # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform
    
    points_per_freq = len(xf) / (xf[-1]) # NOT UNDERSTANDABLE 
    
    slider_range_drum    = st.slider(label='Drum Sound'   , min_value=0, max_value=10, value=1, step=1, key="drum slider")
    slider_range_timpani = st.slider(label='Timpani Sound', min_value=0, max_value=10, value=1, step=1, key="timpani slider")
    slider_range_piccolo = st.slider(label='Piccolo Sound', min_value=0, max_value=10, value=1, step=1, key="piccolo slider")

    yf[int(points_per_freq*0)   :int(points_per_freq* 1000)] *= slider_range_drum
    yf[int(points_per_freq*1000):int(points_per_freq* 2600)] *= slider_range_timpani
    yf[int(points_per_freq*2700):int(points_per_freq*16000)] *= slider_range_piccolo

    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders 
    modified_signal_channel = np.int16(modified_signal) # returns two channels

    write   ("Equalized_Music.wav", sample_rate, modified_signal_channel) # creates the modified song
    st.audio("Equalized_Music.wav", format='audio/wav')

#-------------------------------------- Fourier Transform on Audio ----------------------------------------------------
def general_audio_tranform(audio_file):

    st.audio(audio_file, format='audio/wav') # displaying the audio
    obj = wave.open(audio_file, 'rb')
    sample_rate = obj.getframerate()      # number of samples per second
    n_samples   = obj.getnframes()        # total number of samples in the whole audio
    duration    = n_samples / sample_rate # duration of the audio file
    signal_wave = obj.readframes(-1)      # amplitude of the sound

    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int16)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))

    yf = rfft(signal_y_axis) # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform

    # fig0 = plt.figure(figsize=[14,5])
    # plt.plot(xf, abs(yf))
    # st.plotly_chart(fig0)
    
    points_per_freq = len(xf) / (xf[-1]) # NOT UNDERSTANDABLE 
    
    slider_range_drum    = st.slider(label='Drum Sound'   , min_value=0, max_value=10, value=1, step=1, key="drum slider")
    slider_range_timpani = st.slider(label='Timpani Sound', min_value=0, max_value=10, value=1, step=1, key="timpani slider")
    slider_range_piccolo = st.slider(label='Piccolo Sound', min_value=0, max_value=10, value=1, step=1, key="piccolo slider")

    yf[int(points_per_freq*0)   :int(points_per_freq* 1000)] *= slider_range_drum
    yf[int(points_per_freq*1000):int(points_per_freq* 2600)] *= slider_range_timpani
    yf[int(points_per_freq*2700):int(points_per_freq*16000)] *= slider_range_piccolo

    # fig, axs = plt.subplots()
    # fig.set_size_inches(14,5)
    # plt.plot(xf, np.abs(yf)) #plotting fourier
    # st.plotly_chart(fig)

    modified_signal = irfft(yf) # returning the inverse transform after modifying it with sliders 
    tryyy = np.int16(modified_signal)

    write("example.wav", sample_rate, tryyy)
    st.audio("example.wav", format='audio/wav')

#-------------------------------------- Fourier Transform on Audio ----------------------------------------------------
def arrhythima(dataframe):

    signal_x_axis = (dataframe.iloc[:,0]).to_numpy() # dataframe x axis
    signal_y_axis = (dataframe.iloc[:,1]).to_numpy() # dataframe y axis

    duration    = signal_x_axis[-1]           # the last point in the x axis (number of seconds in the data frame)
    sample_rate = len(signal_y_axis)/duration # returns number points per second

    fourier_x_axis = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform
    fourier_y_axis = rfft(signal_y_axis)                                               # returns complex numbers of the y axis in the data frame

    

    # fig, axs = plt.subplots()
    # fig.set_size_inches(14,5)
    # plt.plot(fourier_x_axis, np.abs(fourier_y_axis)) #plotting fourier
    # st.plotly_chart(fig)
    
    import matplotlib.pyplot as plt
    import numpy as np
    ecg = electrocardiogram()


    fs = 360
    time = np.arange(ecg.size) / fs
    ndex_drums = np.where((time >= 47.2) & (time < 47.8))

    ecg_amps =[]
    time_amps = []
    for i in ndex_drums:
        ecg_amps.append(ecg[i])
        time_amps.append(time[i])

    st.write(list(time_amps))

    fs = 1
    time_hot = np.arange(len(ecg_amps)) / fs

    fourier_x_axis_hot = rfftfreq(len(ecg_amps), (time_amps[1]-time_amps[0]))
    fourier_y_axis_hot = rfft(ecg_amps)

    fig0 = plt.figure(figsize=[9,5])
    plt.plot(fourier_x_axis_hot, abs(fourier_y_axis_hot))
    st.plotly_chart(fig0)  

    fourier_x_axis = rfftfreq(len(ecg), (time[1]-time[0]))
    fourier_y_axis = rfft(ecg)


    points_per_freq = len(fourier_x_axis) / (fourier_x_axis[-1])

    fourier_y_axis[int(points_per_freq*100)   :int(points_per_freq* 150)] *= 0

    modified_signal         = irfft(fourier_y_axis) 

    fig0 = plt.figure(figsize=[9,5])
    plt.plot(fourier_x_axis, abs(fourier_y_axis))
    st.plotly_chart(fig0)  

    fig, axs = plt.subplots()
    fig.set_size_inches(14,5)

    plt.plot(time, (modified_signal))
    plt.xlabel("time in s")
    plt.ylabel("ECG in mV")
    plt.xlim(47, 48)

    st.plotly_chart(fig)
