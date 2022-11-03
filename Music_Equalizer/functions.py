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

#  ----------------------------------- TRIAL FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def dataframe_fourier_transform(dataframe):

    signal_x_axis = (dataframe.iloc[:,0]).to_numpy() # dataframe x axis
    signal_y_axis = (dataframe.iloc[:,1]).to_numpy() # dataframe y axis

    duration    = signal_x_axis[-1] # the last point in the x axis (number of seconds in the data frame)
    sample_rate = len(signal_y_axis)/duration # returns number points per second

    fourier_x_axis = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform
    fourier_y_axis = rfft(signal_y_axis) # returns complex numbers of the y axis in the data frame

    peaks = find_peaks(fourier_y_axis) # computes peaks of the signal 
    peaks_indeces = peaks[0]  # list of indeces of frequency with high peaks

    points_per_freq = len(fourier_x_axis) / (sample_rate) # NOT UNDERSTANDABLE 
    
    fourier_y_axis = dataframe_creating_sliders(peaks_indeces, points_per_freq, fourier_x_axis, fourier_y_axis) # calling creating sliders function

    dataframe_fourier_inverse_transform(fourier_y_axis,signal_x_axis)

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
            slider_range = svs.vertical_slider(min_value=0.0, max_value=2.0, default_value=1., step=.1, key=index)
            st.write(type((slider_range)))
        # these three lines determine the range that will be modified by the slider
        # needs more investigation
        minimum_range = int(points_per_freq * (peaks_indeces[index] - 1.5))
        maximum_range = int(points_per_freq * (peaks_indeces[index] + 1.5))
        fourier_y_axis[minimum_range - 1 : maximum_range + 2] *= slider_range
    return fourier_y_axis

#-------------------------------------- Fourier Transform on Audio ----------------------------------------------------
def audio_fourier_transform(audio_file):

    st.audio(audio_file, format='audio/wav') # displaying the audio
    obj = wave.open(audio_file, 'rb')
    sample_rate = obj.getframerate()   # number of samples per second
    n_samples = obj.getnframes()       # total number of samples in the whole audio
    duration = n_samples / sample_rate # duration of the audio file
    signal_wave = obj.readframes(-1)   # amplitude of the sound
    st.write(signal_wave)
    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int32)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))

    # plotting original audio file
    fig0 = plt.figure(figsize=[10,6])
    plt.plot(signal_x_axis, signal_y_axis)
    st.pyplot(fig0)

    yf = rfft(signal_y_axis) # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform

    peaks = find_peaks(yf)   # computes peaks of the signal 
    peaks_indeces = peaks[0] # indeces of frequency with high peaks
    points_per_freq = len(xf) / (sample_rate) # NOT UNDERSTANDABLE 
    
    slider_range = st.slider(label='hehe', min_value=0.0, max_value=2.0, value=1.0, step=.1)
    
    # these three lines determine the range that will be modified by the slider
    target_idx   = int(points_per_freq * (peaks_indeces[100])) 
    target_idx_2 = int(points_per_freq * (peaks_indeces[300]+2000))
    yf[target_idx - 100 : target_idx_2 + 200] *= slider_range

    fig, axs = plt.subplots()
    fig.set_size_inches(14,5)
    plt.plot(xf, np.abs(yf)) #plotting fourier
    st.pyplot(fig)

    modified_signal = irfft(yf) # returning the inverse transform after modifying it with sliders 
    tryyy = np.int32(modified_signal)

    write("example.wav", sample_rate, tryyy)
    st.audio("example.wav", format='audio/wav')

#  ----------------------------------- JUST  A REFRENCE CODE TO HELP WHILE CREATING SLIDER ---------------------------------------------------------------
def creating_sliders(names_list):

    # Side note: we can change sliders colors and can customize sliders as well.
    # names_list = [('Megzawy', 100),('Magdy', 150)]
    columns = st.columns(len(names_list))
    boundary = int(50)
    sliders_values = []
    sliders = {}

    for index, tuple in enumerate(names_list): # ---> [ { 0, ('Megzawy', 100) } , { 1 , ('Magdy', 150) } ]
        # st.write(index)
        # st.write(i)
        min_value = tuple[1] - boundary
        max_value = tuple[1] + boundary
        key = f'member{random.randint(0,10000000000)}'
        with columns[index]:
            sliders[f'slidergroup{key}'] = svs.vertical_slider(key=key, default_value=tuple[1], step=1, min_value=min_value, max_value=max_value)
            if sliders[f'slidergroup{key}'] == None:
                sliders[f'slidergroup{key}'] = tuple[1]
            sliders_values.append((tuple[0], sliders[f'slidergroup{key}']))
# names_list = [('A', 100),('B', 150),('C', 75),('D', 25),('E', 150),('F', 60),('G', 86),('H', 150),('E', 150),('G', 25),('K', 99),('L', 150),
#                 ('M', 150),('M', 55),('N', 150)]
# fn.creating_sliders(names_list)