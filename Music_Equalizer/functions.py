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
import librosa
import librosa.display
import IPython.display as ipd
import os
import streamlit.components.v1 as components
from scipy import signal
#import time
import altair as alt
import pandas as pd

class Variables:

    points_num=1000
    start=0
    vowel_freq_ae=[860,2850]
    vowel_freq_a=[850,2800]
    slider_tuple=(vowel_freq_ae,vowel_freq_a)

#-------------------------------------- CUSTOM SLIDER ----------------------------------------------------
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "build")
_vertical_slider = components.declare_component("vertical_slider", path=build_dir)

def vertical_slider(key=None):
    slider_value = _vertical_slider(key=key ,default=1)
    return slider_value

#-------------------------------------- WINDOW FUNCTION FOR VOWELS--------------------------------------
def triangle_window(y_fourier, start, end, val, points_per_freq ):
    target_freq = y_fourier[int(start* points_per_freq):int(end*points_per_freq)]
    if val==0:
        window = -(signal.windows.triang(len(target_freq))-1)
    elif val ==1:
            return target_freq 
    else:
        window= val* signal.windows.triang(len(target_freq))
    return [target_freq[i]*window[i] for i in range(len(window))]    

#-------------------------------------- MEDICAL APPLICATION ----------------------------------------------------
def arrhythima( column2, column3):
    ecg_dataset        = electrocardiogram()                                        # Calling the arrhythmia database of a woman
    sampling_frequency = 360                                                        # determining f sample
    time               = np.arange(ecg_dataset.size) / sampling_frequency           # detrmining time axis

    y_fourier, points_per_freq = fourier_transform(ecg_dataset, sampling_frequency) # Fourier Transfrom

    sliders_labels = 'Arrhythima' 

    y_fourier      = f_ranges(y_fourier, points_per_freq, 1, sliders_labels, "Arrhythima")

    plotting_graphs("Original ecg_dataset", column2, time, ecg_dataset, True)

    modified_signal = irfft(y_fourier) 

    plotting_graphs("Modified ecg_dataset", column3, time, modified_signal, True)

#-------------------------------------- VOICE TONE CHANGER ----------------------------------------------------
def voice_changer(uploaded_file, column1, column2):

    signal_x_axis, signal_y_axis, sample_rate = read_audio(uploaded_file)

    plotting_graphs  ('original', column2, signal_x_axis, signal_y_axis, False)

    voice = column1.radio('Voice', options=["Deep Voice", "Smooth Voice"])

    column2.audio(uploaded_file, format="audio/wav")

    if voice == "Deep Voice":
        empty = column2.empty()
        empty.empty()
        speed_rate           = 1.4
        sampling_rate_factor = 1.4

    elif voice == "Smooth Voice":
        empty = column2.empty()
        empty.empty()
        speed_rate           = 0.5
        sampling_rate_factor = 0.5

    loaded_sound_file, sampling_rate = librosa.load(uploaded_file, sr=None)
    loaded_sound_file                = librosa.effects.time_stretch(loaded_sound_file, rate=speed_rate)

    song = ipd.Audio(loaded_sound_file, rate = sampling_rate / sampling_rate_factor)
    empty.write(song)
    
#-------------------------------------- FOURIER TRANSFORM ----------------------------------------------------
def fourier_transform(signal_y_axis, sample_rate):
    y_fourier       = rfft(signal_y_axis)                                # returns complex numbers of the y axis in the data frame
    x_fourier       = rfftfreq(len(signal_y_axis), 1/sample_rate)        # returns the frequency x axis after fourier transform
    points_per_freq = len(x_fourier) / (x_fourier[-1])                   # duration
    return y_fourier, points_per_freq

#-------------------------------------- CREATE SLIDERS & MODIFY SIGNALS ----------------------------------------------------
def f_ranges(y_fourier,points_per_freq,n_sliders,sliders_labels,mode):

    columns = st.columns(n_sliders)
    counter = 0
    list_of_sliders_values = []
    while counter < n_sliders:
        with columns[counter]:
            st.write(sliders_labels[counter])
            sliders = (vertical_slider(counter))
        counter += 1
        list_of_sliders_values.append(sliders)

    if   mode == "Default":
        for index,value in enumerate(list_of_sliders_values):
            y_fourier[int(points_per_freq * 1000 * index)  : int(points_per_freq * 1000 * index + points_per_freq * 1000)] *= value

    elif mode == "Music"  :
        y_fourier[                         :int(points_per_freq* 1000)] *= list_of_sliders_values[0] * .35
        y_fourier[int(points_per_freq*1000):int(points_per_freq* 2600)] *= list_of_sliders_values[1]
        y_fourier[int(points_per_freq*2600):                          ] *= list_of_sliders_values[2] * 0.6

    elif mode == "Vowels" :
        # sliders_labels = ['Z','/i:/','/e/','ʊə','F']

        #  Z ranges
        y_fourier[int(130*points_per_freq):int(240*points_per_freq)] = triangle_window(y_fourier, 130,240, list_of_sliders_values[0], points_per_freq) 
        y_fourier[int(350*points_per_freq):int(470*points_per_freq)] =  triangle_window(y_fourier, 350,470, list_of_sliders_values[0], points_per_freq) [0]
        y_fourier[int(260*points_per_freq):int(350*points_per_freq)] =   triangle_window(y_fourier, 260,350, list_of_sliders_values[0], points_per_freq) 
        y_fourier[int(8000*points_per_freq):int(14000*points_per_freq)] = triangle_window(y_fourier, 8000,14000, list_of_sliders_values[0], points_per_freq) 
        #/i:/ ranges
        y_fourier[int(280*points_per_freq):int(360*points_per_freq)]  = triangle_window(y_fourier, 280,360, list_of_sliders_values[1], points_per_freq) 
        y_fourier[int(210*points_per_freq):int(280*points_per_freq)]  = triangle_window(y_fourier, 210,280, list_of_sliders_values[1], points_per_freq)
        y_fourier[int(130*points_per_freq):int(210*points_per_freq)]  = triangle_window(y_fourier, 130,210, list_of_sliders_values[1], points_per_freq)
        y_fourier[int(340*points_per_freq):int(470*points_per_freq)]  = triangle_window(y_fourier, 340,470, list_of_sliders_values[1], points_per_freq)
        y_fourier[int(3000*points_per_freq):int(3800*points_per_freq)]  = triangle_window(y_fourier, 3000,3800, list_of_sliders_values[1], points_per_freq)
        y_fourier[int(5000*points_per_freq):int(6300*points_per_freq)]  = triangle_window(y_fourier, 5000,6300, list_of_sliders_values[1], points_per_freq)
        # /e/ ranges
        #for e 
        y_fourier[int(342*points_per_freq):int(365*points_per_freq)]  = triangle_window(y_fourier, 342,365, list_of_sliders_values[2], points_per_freq)
        y_fourier[int(310*points_per_freq):int(330*points_per_freq)] = triangle_window(y_fourier, 310,330, list_of_sliders_values[2], points_per_freq)
        # y_fourier[int(170*points_per_freq):int(250*points_per_freq)] = triangle_window(y_fourier, 130,240, list_of_sliders_values[0], points_per_freq)
        # y_fourier[int(685*points_per_freq):int(695*points_per_freq)] = triangle_window(y_fourier, 130,240, list_of_sliders_values[0], points_per_freq)
        # y_fourier[int(702*points_per_freq):int(720*points_per_freq)]  = triangle_window(y_fourier, 130,240, list_of_sliders_values[0], points_per_freq)
        # y_fourier[int(840*points_per_freq):int(1100*points_per_freq)]  = triangle_window(y_fourier, 130,240, list_of_sliders_values[0], points_per_freq)
        #/ʊə/ ranges
        #HAVEN'T BEEN DETECTED YET
        y_fourier[int(2980*points_per_freq):int(3670*points_per_freq)]  = triangle_window(y_fourier, 2980,3670, list_of_sliders_values[3], points_per_freq)
        # y_fourier[int(3670*points_per_freq):int(4740*points_per_freq)]   = triangle_window(y_fourier, 130,240, list_of_sliders_values[0], points_per_freq)
        y_fourier[int(140*points_per_freq):int(308*points_per_freq)]  = triangle_window(y_fourier, 140,308, list_of_sliders_values[3], points_per_freq)
        y_fourier[int(320*points_per_freq):int(370*points_per_freq)]  = triangle_window(y_fourier, 320,370, list_of_sliders_values[3], points_per_freq)
        #F ranges
        #HAVEN'T BEEN DETECTED YET
        y_fourier[int(2980*points_per_freq):int(3670*points_per_freq)]  = triangle_window(y_fourier, 2980,3670, list_of_sliders_values[4], points_per_freq)
        # y_fourier[int(3670*points_per_freq):int(4740*points_per_freq)]  *= list_of_sliders_values[4]  
        # y_fourier[int(140*points_per_freq):int(308*points_per_freq)] *= list_of_sliders_values[4] 
        # y_fourier[int(320*points_per_freq):int(370*points_per_freq)] *= list_of_sliders_values[4] 

    elif mode == "Arrhythima":
        y_fourier[int(points_per_freq*0) : int(points_per_freq* 5)] *= list_of_sliders_values[0] 

    elif mode == "Voice Tone Changer":
        pass

    return y_fourier

#-------------------------------------- READ AUDIO FILES ----------------------------------------------------
def read_audio(audio_file):
    obj = wave.open(audio_file, 'rb')                            # creating object
    sample_rate   = obj.getframerate()                           # number of samples per second
    n_samples     = obj.getnframes()                             # total number of samples in the whole audio
    signal_wave   = obj.readframes(-1)                           # amplitude of the sound
    duration      = n_samples / sample_rate                      # duration of the audio file
    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int16)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))
    return signal_x_axis, signal_y_axis, sample_rate

#-------------------------------------- PLOTTING SPECTROGRAM ----------------------------------------------------
def plot_spectro(original_audio, modified_audio):
    
    y1, sr = librosa.load(original_audio)
    y2, sr = librosa.load(modified_audio)
    D1     = librosa.stft(y1)             # STFT of y
    S_db1  = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
    D2     = librosa.stft(y2)             # STFT of y
    S_db2  = librosa.amplitude_to_db(np.abs(D2), ref=np.max)

    fig= plt.figure(figsize=[15,10])
    plt.subplot(2,2,1)
    img1 = librosa.display.specshow(S_db1, x_axis='time', y_axis='linear')
    plt.subplot(2,2,2)
    img2 = librosa.display.specshow(S_db2, x_axis='time', y_axis='linear')
    # fig.colorbar(img1, format="%+2.f dB")
    # fig.colorbar(img2, format="%+2.f dB")

    st.pyplot(fig)

#-------------------------------------- DYNAMIC PLOTTING ----------------------------------------------------
def plot_animation(df):
    brush  = alt.selection_interval ()
    chart1 = alt.Chart(df).mark_line().encode(x=alt.X('time', axis=alt.Axis(title='Time')),).properties(width=414,height=250).add_selection(brush).interactive()
    figure = chart1.encode(y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude'))) | chart1.encode(y ='amplitude after processing').add_selection(brush)
    return figure

def Dynamic_graph(signal_x_axis, signal_y_axis, signal_y_axis1,start_btn,pause_btn,resume_btn):

        df = pd.DataFrame({'time': signal_x_axis[::200], 'amplitude': signal_y_axis[:: 200], 'amplitude after processing': signal_y_axis1[::200]}, columns=['time', 'amplitude','amplitude after processing'])

        lines       = plot_animation(df)
        line_plot   = st.altair_chart(lines)

        df_elements = df.shape[0] # number of elements in the dataframe
        burst       = 10          # number of elements  to add to the plot
        size        = burst       # size of the current dataset

        if start_btn:
            for i in range(1, df_elements):
                Variables.start      = i
                step_df              = df.iloc[0:size]
                lines                = plot_animation(step_df)
                line_plot            = line_plot.altair_chart(lines)
                Variables.graph_size = size
                size                 = i * burst 

        if resume_btn: 
            for i in range(Variables.start,df_elements):
                Variables.start      = i
                step_df              = df.iloc[0:size]
                lines                = plot_animation(step_df)
                line_plot            = line_plot.altair_chart(lines)
                Variables.graph_size = size
                size                 = i * burst

        if pause_btn:
            step_df   = df.iloc[0:Variables.graph_size]
            lines     = plot_animation(step_df)
            line_plot = line_plot.altair_chart(lines)

#--------------------------------------  STATIC PLOTTING ----------------------------------------------------
def plotting_graphs(title, column, x_axis, y_axis, flag):
    fig, axs = plt.subplots()
    fig.set_size_inches(6,3)
    plt.plot(x_axis,y_axis)
    plt.title(title)
    if flag == True:
        plt.xlim(45, 55)
        plt.xlabel("Time in s")
        plt.ylabel("ecg_dataset in mV")
    column.pyplot(fig)