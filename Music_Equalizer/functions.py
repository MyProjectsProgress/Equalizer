import random 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import streamlit_vertical_slider  as svs
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy.fft import fft, fftfreq, fftshift
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io.wavfile import write
from scipy.signal import find_peaks

#  ----------------------------------- TRIAL FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def dataframe_fourier_transform(dataframe):

    signal_y_axis = (dataframe.iloc[:,1]).to_numpy() # dataframe y axis
    signal_x_axis = (dataframe.iloc[:,0]).to_numpy() # dataframe x axis

    duration    = signal_x_axis[-1] # the last point in the x axis (number of seconds in the data frame)
    sample_rate = len(signal_y_axis) / duration # returns number points per second

    yf = rfft(signal_y_axis) # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform

    peaks = find_peaks(yf) # computes peaks of the signal 

    peaks_indeces = peaks[0] # indeces of frequency with high peaks

    points_per_freq = len(xf) / (sample_rate) # NOT UNDERSTANDABLE 

    slider_range = st.slider(label='hehe', min_value=0.0, max_value=2.0, value=.1, step=.1)

    # these three lines determine the range that will be modified by the slider
    target_idx   = int(points_per_freq * (peaks_indeces[1]-1)) 
    target_idx_2 = int(points_per_freq * (peaks_indeces[1]+1))
    yf[target_idx - 1 : target_idx_2 + 2] *= slider_range

    modified_signal = irfft(yf) # returning the inverse transform after modifying it with sliders 

    fig, axs = plt.subplots()
    fig.set_size_inches(14,5)
    
    plt.plot(xf, np.abs(yf)) #plotting signal before modifying
    plt.plot(xf[peaks_indeces[:]], np.abs(yf)[peaks_indeces[:]], marker="o") # plotting peaks points
    st.plotly_chart(fig,use_container_width=True)

    fig2, axs2 = plt.subplots()
    fig2.set_size_inches(14,5)
    plt.plot(signal_x_axis,modified_signal) # ploting signal after modifying
    st.plotly_chart(fig2,use_container_width=True)

#  ----------------------------------- INVERSE FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def fourier_inverse_transform(inverse_fourier,df):
    # Getting df x_axis and y_axis
    list_of_columns = df.columns
    df_x_axis = list(df[list_of_columns[0]])
    df_y_axis = list(df[list_of_columns[1]])

    # Create subplot
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)

    # Frequency domain representation
    axis.set_title('Inverse Fourier transform depicting the frequency components')
    axis.plot(df_x_axis, inverse_fourier)
    axis.set_xlabel('Frequency')
    axis.set_ylabel('Amplitude')

    fig,ax = plt.subplots()
    ax.set_title('The Actual Data')
    ax.plot(df_x_axis,df_y_axis)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')

    st.plotly_chart(figure,use_container_width=True)
    st.plotly_chart(fig,use_container_width=True)

def wave_ranges(fourier_transform):
    st.write(abs(fourier_transform))

#  ----------------------------------- CREATING SLIDERS ---------------------------------------------------------------
def creating_sliders(names_list):

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
            # st.write(sliders_values)

#  ----------------------------------- PLOTTING AUDIO ---------------------------------------------------------------
# def plotting(amplitude, frequency):
#     # Plotting audio Signal
#     fig = plt.figure(figsize=[10,6])
#     librosa.display.waveshow(amplitude, frequency)
#     st.pyplot(fig)

#-------------------------------------- Fourier Transform on Audio ----------------------------------------------------
def fourier_for_audio(uploaded_file):
    sample_rate, amplitude = wav.read(uploaded_file)  # kam sample fl sec fl track,amplitude l data
    amplitude = np.frombuffer(amplitude, "int32")     # str code khd mn dof3t 4 - 3 ayam search
    fft_out = fft(amplitude)                          # el fourier bt3t el amplitude eli hnshtghl beha fl equalizer
    fft_out = np.abs(fft_out)[:len(amplitude)//2]     # np.abs 3shan el rsm
    # plt.plot(amplitude, np.abs(fft_out))
    # plt.show() satren code mbyrsmosh haga 
    x_axis_fourier = fftfreq(len(amplitude),(1/sample_rate))[:len(amplitude)//2] #3shan mbd2sh mn -ve
    return x_axis_fourier,fft_out

def plotting(x_axis_fourier,fft_out):
    # Plotting audio Signal
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)
    axis.plot(x_axis_fourier,fft_out)
    st.plotly_chart(figure,use_container_width=True) 
#-------------tagroba fashla------------
