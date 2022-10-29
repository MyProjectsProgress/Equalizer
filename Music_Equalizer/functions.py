import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------------------------Signal Object
class Audio_Sliders:
    def __init__(self,slider_name,frequency_mini_value, frequency_max_value):
        self.amplitude = st.slider(f'{slider_name}', min_value=frequency_mini_value, max_value=frequency_max_value, step=1, key=f'{slider_name}') 

class Music_Sliders:
    def __init__(self,slider_name,frequency_mini_value, frequency_max_value):
        self.amplitude = st.slider(f'{slider_name}', min_value=frequency_mini_value, max_value=frequency_max_value, step=1, key=f'{slider_name}') 

class Vowels_Sliders:
    def __init__(self,slider_name,frequency_mini_value, frequency_max_value):
        self.amplitude = st.slider(f'{slider_name}', min_value=frequency_mini_value, max_value=frequency_max_value, step=1, key=f'{slider_name}') 

class Medical_Sliders:
    def __init__(self,slider_name,frequency_mini_value, frequency_max_value):
        self.amplitude = st.slider(f'{slider_name}', min_value=frequency_mini_value, max_value=frequency_max_value, step=1, key=f'{slider_name}') 

#  ----------------------------------- FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def fourier_transform(df):
    # Getting df x_axis and y_axis
    list_of_columns = df.columns
    df_x_axis = (df[list_of_columns[0]])
    df_y_axis = (df[list_of_columns[1]])

    # Frequency domain representation
    fourierTransform = np.fft.fft(df_y_axis)

    # Do an inverse Fourier transform on the signal
    inverseFourier = np.fft.ifft(fourierTransform)

    # Create subplot
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)

    # Frequency domain representation
    axis.set_title('Fourier transform depicting the frequency components')
    axis.plot(df_x_axis, abs(fourierTransform))
    axis.set_xlabel('Frequency')
    axis.set_ylabel('Amplitude')

    st.plotly_chart(figure,use_container_width=True)

    return inverseFourier, fourierTransform

#  ----------------------------------- INVERSE FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def fourier_inverse_transform(inverseFourier,df):
    # Getting df x_axis and y_axis
    list_of_columns = df.columns
    df_x_axis = list(df[list_of_columns[0]])
    df_y_axis = list(df[list_of_columns[1]])

    # Create subplot
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)

    # Frequency domain representation
    axis.set_title('Inverse Fourier transform depicting the frequency components')
    axis.plot(df_x_axis, inverseFourier)
    axis.set_xlabel('Frequency')
    axis.set_ylabel('Amplitude')

    fig,ax = plt.subplots()
    ax.set_title('The Actual Data')
    ax.plot(df_x_axis,df_y_axis)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')

    st.plotly_chart(figure,use_container_width=True)
    st.plotly_chart(fig,use_container_width=True)

def wave_ranges(fourierTransform):
    st.write(abs(fourierTransform))