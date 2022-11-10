import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import streamlit_vertical_slider  as svs
# from scipy.misc import electrocardiogram
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io.wavfile import write
import wave
import librosa
import librosa.display
import IPython.display as ipd
import os
import streamlit.components.v1 as components

#-------------------------------------- Custom Slider ----------------------------------------------------
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "build")
_vertical_slider = components.declare_component("vertical_slider", path=build_dir)

def vertical_slider(key=None):
    slider_value = _vertical_slider(key=key ,default=1)
    return slider_value

#-------------------------------------- MEDICAL APPLICATION ----------------------------------------------------
def arrhythima(column1, column2, column3, show_spectro, dataframe):
    y1 = dataframe.iloc[12,:290]
    x1 = range(len(y1))
    duration=2
    signal_y_axis=y1.values.flatten() 
    signal_x_axis=np.array(x1)
    plotting_graphs('original',column2,signal_x_axis,signal_y_axis, False)
    sample_rate=len(signal_y_axis)/duration
    yf, points_per_freq=fourier_transform(signal_y_axis, sample_rate)
    xf = rfftfreq(len(signal_y_axis), 1/sample_rate) 
    points_per_freq = len(xf) / (xf[-1])

    sliders_labels = ['1', '2']
    yf = equalizer(yf, points_per_freq,2, sliders_labels, "arrhythmia")

    modified_signal         = irfft(yf)
    plotting_graphs('Modified',column2,signal_x_axis,modified_signal,False)



    # y2 = dataframe.iloc[9,:290]
    # x2 = range(len(y2))
    # duration=2

    # signal_y_axis=y2.values.flatten() 
    # signal_x_axis=np.array(x2)
    # plotting_graphs(column3,signal_x_axis,signal_y_axis, False)
    # sample_rate=len(signal_y_axis)/duration
    # yf, points_per_freq=fourier_transform(signal_y_axis, sample_rate)
    # xf = rfftfreq(len(signal_y_axis), 1/sample_rate) 
    # # points_per_freq = len(xf) / (xf[-1])
    # # yf[int(points_per_freq*0)   :int(points_per_freq* 5.5)] *= 0
    # sliders_labels = ['1', '2']
    # yf = equalizer(yf, points_per_freq,2, sliders_labels, "arrhythmia")
    # modified_signal         = irfft(yf)
    # plotting_graphs(column3,signal_x_axis,modified_signal,False)


    # # normal beat
    # y1 = dataframe.iloc[0,:186]
    # x1 = range(len(y1))
    # plotting_graphs(column2, x1, y1, False)

    # # unknown beat
    # y2 = dataframe.iloc[1,:186]
    # x2 = range(len(y2))
    # plotting_graphs(column3, x2, y2, False)

    # # ventriculer etopic beat
    # y3 = dataframe.iloc[2,:186]
    # x3 = range(len(y3))
    # plotting_graphs(column2, x3, y3, False)

    # # super ventriculer etopic beat
    # y4 = dataframe.iloc[3,:186]
    # x4 = range(len(y4))
    # plotting_graphs(column3, x4, y4, False)

    # # fusion beat
    # y5 = dataframe.iloc[4,:186]
    # x5 = range(len(y5))
    # plotting_graphs(column2, x5, y5, False)

#-------------------------------------- OPTIONAL ----------------------------------------------------
def voice_changer(uploaded_file, column1, column2, column3, show_spectro):

    signal_x_axis, signal_y_axis, sample_rate = read_audio(uploaded_file)

    if (show_spectro):
        plot_spectro('original',column2, uploaded_file.name)
    else:
        plotting_graphs('original',column2,signal_x_axis,signal_y_axis,False)

    voice = column1.radio('Voice', options=["Deep Voice", "Smooth Voice"])

    column2.audio(uploaded_file, format="audio/wav")

    if voice == "Deep Voice":
        empty = column2.empty()
        empty.empty()
        speed_rate = 1.4
        sampling_rate_factor = 1.4

    elif voice == "Smooth Voice":
        empty = column2.empty()
        empty.empty()
        speed_rate = 0.5
        sampling_rate_factor = 0.5

    loaded_sound_file, sampling_rate = librosa.load(uploaded_file, sr=None)
    loaded_sound_file                = librosa.effects.time_stretch(loaded_sound_file, rate=speed_rate)

    # if (show_spectro):
    #     plot_spectro(column2, uploaded_file.name)
    # else:
    #     plotting_graphs(column2,signal_x_axis,signal_y_axis,False)

    song = ipd.Audio(loaded_sound_file, rate = sampling_rate / sampling_rate_factor)
    empty.write(song)

#-------------------------------------- Fourier Transfrom ----------------------------------------------------
def fourier_transform(signal_y_axis, sample_rate):
    yf = rfft(signal_y_axis)                                # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), 1/sample_rate)        # returns the frequency x axis after fourier transform
    points_per_freq = len(xf) / (xf[-1])                    # duration
    return yf, points_per_freq

#-------------------------------------- Create Sliders & Modify Signal ----------------------------------------------------
def equalizer(yf,points_per_freq,n_sliders,sliders_labels,mode):

    columns=st.columns(n_sliders)
    counter=0
    list_of_sliders_values = []
    while counter < n_sliders:
        with columns[counter]:
            st.write(sliders_labels[counter])
            sliders = (vertical_slider(counter))
        counter +=1
        list_of_sliders_values.append(sliders)

    if mode == "Default":
        for index,value in enumerate(list_of_sliders_values):
            yf[int(points_per_freq * 1000 * index)  : int(points_per_freq * 1000 * index + points_per_freq * 1000)] *= value

    elif mode == "Music":
        yf[:int(points_per_freq* 1000)] *= list_of_sliders_values[0] * .35
        yf[int(points_per_freq*1000):int(points_per_freq* 2600)] *= list_of_sliders_values[1] 
        yf[int(points_per_freq*2600):] *= list_of_sliders_values[2] * 0.6

    elif mode == "Vowels":
        # sliders_labels = ['Z','/i:/','/e/','ʊə','F']

        #  Z ranges
        yf[int(130*points_per_freq):int(240*points_per_freq)] *=  list_of_sliders_values[0]
        yf[int(350*points_per_freq):int(470*points_per_freq)] *=  list_of_sliders_values[0]
        yf[int(260*points_per_freq):int(350*points_per_freq)] *=  list_of_sliders_values[0]
        yf[int(8000*points_per_freq):int(14000*points_per_freq)] *= list_of_sliders_values[0]
        #/i:/ ranges
        yf[int(280*points_per_freq):int(360*points_per_freq)] *= list_of_sliders_values[1]
        yf[int(210*points_per_freq):int(280*points_per_freq)] *= list_of_sliders_values[1]
        yf[int(130*points_per_freq):int(210*points_per_freq)] *= list_of_sliders_values[1]
        yf[int(340*points_per_freq):int(470*points_per_freq)] *= list_of_sliders_values[1]
        yf[int(3000*points_per_freq):int(3800*points_per_freq)] *= list_of_sliders_values[1]
        yf[int(5000*points_per_freq):int(6300*points_per_freq)] *= list_of_sliders_values[1]
        # /e/ ranges
        #for e 
        yf[int(342*points_per_freq):int(365*points_per_freq)] *= list_of_sliders_values[2]
        yf[int(310*points_per_freq):int(330*points_per_freq)] *= list_of_sliders_values[2]
        yf[int(170*points_per_freq):int(250*points_per_freq)]*= list_of_sliders_values[2]
        yf[int(685*points_per_freq):int(695*points_per_freq)]*= list_of_sliders_values[2]
        yf[int(702*points_per_freq):int(720*points_per_freq)] *= list_of_sliders_values[2]
        yf[int(840*points_per_freq):int(1100*points_per_freq)] *= list_of_sliders_values[2]
        #/ʊə/ ranges
        #HAVEN'T BEEN DETECTED YET
        yf[int(2980*points_per_freq):int(3670*points_per_freq)] *= list_of_sliders_values[3]
        yf[int(3670*points_per_freq):int(4740*points_per_freq)]  *= list_of_sliders_values[3]  
        yf[int(140*points_per_freq):int(308*points_per_freq)] *= list_of_sliders_values[3] 
        yf[int(320*points_per_freq):int(370*points_per_freq)] *= list_of_sliders_values[3] 
        #F ranges
        #HAVEN'T BEEN DETECTED YET
        yf[int(2980*points_per_freq):int(3670*points_per_freq)] *= list_of_sliders_values[4]
        yf[int(3670*points_per_freq):int(4740*points_per_freq)]  *= list_of_sliders_values[4]  
        yf[int(140*points_per_freq):int(308*points_per_freq)] *= list_of_sliders_values[4] 
        yf[int(320*points_per_freq):int(370*points_per_freq)] *= list_of_sliders_values[4] 


    elif mode == "Arrhythima":
        yf[int(points_per_freq*0) : int(points_per_freq* 3.5)] *= list_of_sliders_values[0] #row9
        yf[int(points_per_freq*0) : int(points_per_freq* 5.5)] *= list_of_sliders_values[1] #row12
        # yf[int(points_per_freq*0) : int(points_per_freq* 3.5)] *= eq_slider

    elif mode == "Optional":
        pass

    return yf

#-------------------------------------- Read Audio Files ----------------------------------------------------
def read_audio(audio_file):
    obj = wave.open(audio_file, 'rb')                            # creating object
    sample_rate   = obj.getframerate()                           # number of samples per second
    n_samples     = obj.getnframes()                             # total number of samples in the whole audio
    signal_wave   = obj.readframes(-1)                           # amplitude of the sound
    duration      = n_samples / sample_rate                      # duration of the audio file
    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int16)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))
    return signal_x_axis, signal_y_axis, sample_rate

#-------------------------------------- PLOTING Time Graph ----------------------------------------------------
def plotting_graphs(title,column,x_axis,y_axis,flag):
    fig, axs = plt.subplots()
    fig.set_size_inches(6,3)
    plt.plot(x_axis,y_axis)
    plt.title(title)
    if flag == True:
        plt.xlim(45, 55)
        plt.xlabel("Time in s")
        plt.ylabel("ECG in mV")
    column.pyplot(fig)

#-------------------------------------- PLOTING Spectrogram ----------------------------------------------------
def plot_spectro(title,column,audio_file):
    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    fig.set_size_inches(6,3)
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.title(title)
    column.pyplot(fig)

    #     st.plotly_chart(fig,use_container_width=True)