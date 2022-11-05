import random 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit_vertical_slider  as svs
from scipy.fftpack import fft
from scipy.fft import fft, fftfreq, fftshift
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io.wavfile import write
from scipy.signal import find_peaks
import wave
import IPython.display as ipd

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

#-------------------------------------- Fourier Transform on Audio ----------------------------------------------------
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

    fig, axs = plt.subplots()
    fig.set_size_inches(14,5)
    plt.plot(xf,np.abs(yf)) #plotting fourier
    st.plotly_chart(fig)



    slider_range_drum    = st.slider(label='Drum Sound'   , min_value=0, max_value=10, value=1, step=1, key="drum slider")
    slider_range_timpani = st.slider(label='Timpani Sound', min_value=0, max_value=10, value=1, step=1, key="timpani slider")
    slider_range_piccolo = st.slider(label='Piccolo Sound', min_value=0, max_value=10, value=1, step=1, key="piccolo slider")

    yf[int(points_per_freq*0)   :int(points_per_freq* 1000)] *= slider_range_drum
  

    yf[int(points_per_freq*1000):int(points_per_freq* 2600)] *= slider_range_timpani
 

    yf[int(points_per_freq*2700):int(points_per_freq*16000)] *= slider_range_piccolo
    fig4, axs2 = plt.subplots()
    fig4.set_size_inches(14,5)
    plt.plot(xf,np.abs(yf)) #plotting fourier
    st.plotly_chart(fig4,use_container_width=True)
    


    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders 
    modified_signal_channel = np.int16(modified_signal) # returns two channels 

    write   ("Equalized_Music.wav", sample_rate, modified_signal_channel) # creates the modified song
    st.audio("Equalized_Music.wav", format='audio/wav')
   
    

#-------------------------------------- Fourier Transform on Audio ----------------------------------------------------
def uniform_range_mode(audio_file):
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
     
    peaks = find_peaks(signal_y_axis) # computes peaks of the signal 
    peaks_indeces = peaks[0]  # list of indeces of frequency with high peaks

    st.write(sample_rate)
    points_per_freq = len(xf) / (xf[-1]) # NOT UNDERSTANDABLE 
    
    fig, axs = plt.subplots()
    fig.set_size_inches(14,5)
    plt.plot(signal_x_axis, signal_y_axis) #plotting fourier
    st.plotly_chart(fig)
    
    col1,col2,col3,col4,col5,col6,col7,col8,col9,col10=st.columns([1,1,1,1,1,1,1,1,1,1])
    with col1:
        slider_range_1 = svs.vertical_slider(key=1,min_value=0, max_value=10, default_value=1, step=1)
        if slider_range_1 is not None:
            yf[int(points_per_freq*0)   :int(points_per_freq* 1000)] *= slider_range_1
    with col2:
        slider_range_2 = svs.vertical_slider(key=2,min_value=0, max_value=10,default_value=1, step=1)
        if slider_range_2 is not None:
            yf[int(points_per_freq*1000):int(points_per_freq* 2000)] *= slider_range_2
    with col3:
        slider_range_3 = svs.vertical_slider(key=3,min_value=0, max_value=10,default_value=1, step=1)
        if slider_range_3 is not None:
            yf[int(points_per_freq*2000):int(points_per_freq*3000)]  *= slider_range_3
    with col4:
        slider_range_4 = svs.vertical_slider(key=4,min_value=0, max_value=10,default_value=1, step=1)
        if slider_range_4 is not None:
            yf[int(points_per_freq*3000):int(points_per_freq*4000)]  *= slider_range_4
    with col5:
        slider_range_5 = svs.vertical_slider(key=5,min_value=0, max_value=10,default_value=1, step=1)
        if slider_range_5 is not None:
            yf[int(points_per_freq*4000):int(points_per_freq*5000)]  *= slider_range_5
    with col6:
        slider_range_6 =  svs.vertical_slider(key=6,min_value=0, max_value=10,default_value=1, step=1)
        if slider_range_6 is not None:
           yf[int(points_per_freq*5000):int(points_per_freq*6000)]  *= slider_range_6
    with col7:
        slider_range_7 = svs.vertical_slider(key=7,min_value=0, max_value=10,default_value=1, step=1)
        if slider_range_7 is not None:
            yf[int(points_per_freq*6000):int(points_per_freq*7000)]  *= slider_range_7
    with col8:
        slider_range_8 = svs.vertical_slider(key=8,min_value=0, max_value=10,default_value=1, step=1)
        if slider_range_8 is not None:
            yf[int(points_per_freq*7000):int(points_per_freq*8000)]  *= slider_range_8
    with col9:
        slider_range_9 = svs.vertical_slider(key=9,min_value=0, max_value=10,default_value=1, step=1)
        if slider_range_9 is not None:
             yf[int(points_per_freq*8000):int(points_per_freq*9000)]  *= slider_range_9
    with col10:
        slider_range_10= svs.vertical_slider(key=10,min_value=0, max_value=10,default_value=1, step=1)
        if slider_range_10 is not None:
            yf[int(points_per_freq*9000):int(points_per_freq*10000)] *= slider_range_10

        
        # fig2, axs2 = plt.subplots()
        # fig2.set_size_inches(14,5)
        # plt.plot(xf,np.abs(yf)) # ploting signal after modifying
        # st.plotly_chart(fig2,use_container_width=True)




    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders 
    modified_signal_channel = np.int16(modified_signal) # returns two channels 

        # yf=audio_creating_sliders(peaks_indeces,points_per_freq,xf,yf)

    write   ("Equalized_Music.wav", sample_rate, modified_signal_channel) # creates the modified song
    st.audio("Equalized_Music.wav", format='audio/wav')
