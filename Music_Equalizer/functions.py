import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
import wave
import librosa
import librosa.display
import os
import streamlit.components.v1 as components
import altair as alt
import pandas as pd
from scipy.io.wavfile import write

#-------------------------------------- VOICE TONE CHANGER ----------------------------------------------------
def voice_changer(uploaded_file, column1, column2):

    voice = column1.radio('Voice', options=["Normal Voice","Deep Voice", "Smooth Voice"])
    
    if voice   == "Normal Voice":
        Num_of_steps = 0

    if voice   == "Deep Voice":
        Num_of_steps = -7

    elif voice == "Smooth Voice":
        Num_of_steps = 7

    signal, sample_rate = librosa.load(uploaded_file, sr=None)
    modified_signal     = librosa.effects.pitch_shift(signal, sr=sample_rate, n_steps=Num_of_steps)
    write("voice_changed.wav", sample_rate, modified_signal)

    time =np.linspace(0,signal.shape[0]/sample_rate,signal.shape[0] ) # read audio file
    
    with column2:
        fig= plt.figure(figsize=[15,8])
        plt.subplot(2,2,1)
        plt.plot(time,signal)
        plt.subplot(2,2,2)
        plt.plot(time,modified_signal)
        st.pyplot(fig)
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
    points_per_freq = len(x_fourier) / (x_fourier[-1])                   # how many points in the x_fourier array from freq 0 to 1 
    return y_fourier, points_per_freq

#-------------------------------------- CREATE SLIDERS & MODIFY SIGNALS ----------------------------------------------------
def f_ranges(y_fourier, points_per_freq, n_sliders, sliders_labels,ranges):

    columns = st.columns(n_sliders)                                     # Create sliders and its' labels
    counter = 0
    list_of_sliders_values = []
    while counter < n_sliders:
        with columns[counter]:
            st.write(sliders_labels[counter])
            sliders = (vertical_slider(counter))
        counter += 1
        list_of_sliders_values.append(sliders)

    for index,value in enumerate(list_of_sliders_values):
        y_fourier[int(ranges[index][0]*points_per_freq):int(ranges[index][1]*points_per_freq)]  *= value

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
        plt.grid   ()

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

    st.pyplot(fig)

#-------------------------------------- DYNAMIC PLOTTING ----------------------------------------------------
class Variables:
    start=0
    size=300

def plot_animation(df):

    # set the interval that will be plotted
    brush  = alt.selection_interval ()

    # customizing the graph x axis, giving names and size of the plot
    chart1 = alt.Chart(df).mark_line().encode(x=alt.X('time', axis=alt.Axis(title='Time',labels=False)),).properties(width=414,height=200).add_selection(brush).interactive()

    # customizing the graph y axis, giving titles and lables
    figure = chart1.encode(y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude'))) | chart1.encode(y ='amplitude after processing').add_selection(brush)
    return figure

def Dynamic_graph(signal_x_axis, signal_y_axis, signal_y_axis1,sample_rate):

        step_plot= int(sample_rate/210)

        # creating the data frame to to work with altair library
        df = pd.DataFrame({'time': signal_x_axis[::step_plot], 'amplitude': signal_y_axis[:: step_plot], 'amplitude after processing': signal_y_axis1[::step_plot]}, columns=['time', 'amplitude','amplitude after processing'])

        lines       = plot_animation(df)     # call plot animation to return the figure "initially all the data frame"
        line_plot   = st.altair_chart(lines) # plotting all the dataframe initially

        df_elements = df.shape[0] # number of elements in the dataframe
        burst       = 300         # number of elements  to add to the plot
        size        = burst       # size of the current dataset

        if (st.session_state.graph_mode == True):
            for i in range(Variables.start, df_elements - burst): # graphs starts from 0 to 300
                Variables.start      = i                          # start is edited ex: [0,1,2,..]
                step_df              = df.iloc[i: burst+i]        # the points we plot are edited ex: (0,300) then (1,301) then (2,302)
                lines                = plot_animation(step_df)    # call plot animation to return the figure
                line_plot.altair_chart(lines)                     # plot the figure
                Variables.graph_size = size                       # modifying the size to be plotted in pause 
                size                 = i * burst                  # edit the size that will be passed to the pause
        else:
            try:
                step_df   = df.iloc[Variables.start : Variables.size]   # start from the constantly edited start variable till the constantly edited size variable
                lines     = plot_animation(step_df)                     # call plot animation to return the figure
                line_plot.altair_chart(lines)                           # plot the figure
            except:
                pass