import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  ----------------------------------- FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def fourier_transform(df):
    # How many time points are needed i,e., Sampling Frequency
    samplingFrequency   = 100

    # At what intervals time points are sampled
    samplingInterval       = 1 / samplingFrequency

    # Getting df x_axis and y_axis
    list_of_columns = df.columns
    df_x_axis = list(df[list_of_columns[0]])
    df_y_axis = list(df[list_of_columns[1]])

    # Begin and End time period of the signals
    beginTime = df[list_of_columns[0]].iat[0] # begin_time
    endTime = df[list_of_columns[0]].iloc[-1] # end time 

    # Frequency domain representation
    fourierTransform = np.fft.fft(df_y_axis)/len(df_y_axis)           # Normalize amplitude
    fourierTransform = fourierTransform[range(int(len(df_y_axis)/2))] # Exclude sampling frequency
    tpCount     = len(df_y_axis)
    values      = np.arange(int(tpCount/2))
    timePeriod  = tpCount/samplingFrequency
    frequencies = values/timePeriod

    # Do an inverse Fourier transform on the signal
    inverseFourier = np.fft.ifft(fourierTransform)

    # Create subplot
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)

    # Frequency domain representation
    axis.set_title('Fourier transform depicting the frequency components')
    axis.plot(frequencies, abs(fourierTransform))
    axis.set_xlabel('Frequency')
    axis.set_ylabel('Amplitude')

    st.plotly_chart(figure,use_container_width=True)

    return inverseFourier

#  ----------------------------------- INVERSE FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def fourier_inverse_transform(inverseFourier,df):
    # Getting df x_axis and y_axis
    list_of_columns = df.columns
    df_x_axis = list(df[list_of_columns[0]])
    df_y_axis = list(df[list_of_columns[1]])
    tpCount     = len(df_y_axis)
    values      = np.arange(int(tpCount/2))

    # Create subplot
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)

    # Frequency domain representation
    axis.set_title('Inverse Fourier transform depicting the frequency components')
    axis.plot(values, inverseFourier)
    axis.set_xlabel('Frequency')
    axis.set_ylabel('Amplitude')

    st.plotly_chart(figure,use_container_width=True)