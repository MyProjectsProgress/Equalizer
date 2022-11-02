import random 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import streamlit_vertical_slider  as svs

#  ----------------------------------- FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def fourier_transform(df):
    # Getting df x_axis and y_axis
    list_of_columns = df.columns
    df_x_axis = (df[list_of_columns[0]])
    df_y_axis = (df[list_of_columns[1]])

    # Frequency domain representation
    fourier_transform = np.fft.fft(df_y_axis)

    # Do an inverse Fourier transform on the signal
    inverse_fourier = np.fft.ifft(fourier_transform)

    # Create subplot
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)

    # Frequency domain representation
    axis.set_title('Fourier transform depicting the frequency components')
    axis.plot(df_x_axis, abs(fourier_transform))
    axis.set_xlabel('Frequency')
    axis.set_ylabel('Amplitude')

    st.plotly_chart(figure,use_container_width=True)

    return inverse_fourier, fourier_transform

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
def plotting(amplitude, frequency):
    # Plotting audio Signal
    fig = plt.figure(figsize=[10,6])
    librosa.display.waveshow(amplitude, frequency)
    st.pyplot(fig)