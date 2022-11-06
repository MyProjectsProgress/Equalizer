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

#-------------------------------------- UNIFORM RANGE MODE ----------------------------------------------------
def uniform_range_mode(column1, column2, column3,show_spectro):

    signal_x_axis, signal_y_axis, sample_rate = read_audio(".piano_timpani_piccolo_out.wav")    # Read Audio File

    yf, points_per_freq = fourier_transform(signal_y_axis, sample_rate)         # Fourier Transfrom

    if (show_spectro):
        plot_spectro(column2,'.piano_timpani_piccolo_out.wav')
    else:
        plotting_graphs(column2,signal_x_axis,signal_y_axis,False)

    sliders_labels = ['0 to 1000 Hz', '1000 to 2000 Hz', '2000 to 3000 Hz','3000 to 4000 Hz',
    '4000 to 5000 Hz', '5000 to 6000 Hz','6000 to 7000 Hz', '7000 to 8000 Hz', '8000 to 9000 Hz','9000 to 10000 Hz']

    yf = equalizer(yf, points_per_freq, 10, sliders_labels, "Default")      #create sliders and modify signal

    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders 
    modified_signal_channel = np.int16(modified_signal) # returns two channels

    write(".Equalized_Music.wav", sample_rate, modified_signal_channel) # creates the modified song

    if (show_spectro):
        plot_spectro(column3,".Equalized_Music.wav")
    else:
        plotting_graphs(column3,signal_x_axis,modified_signal,False)

    column2.audio(".piano_timpani_piccolo_out.wav", format='audio/wav') # displaying the audio before editing
    column3.audio(".Equalized_Music.wav"          , format='audio/wav') # displaying the audio after editing

#-------------------------------------- MUSICAL INSTRUMENTS EQUALIZER ----------------------------------------------------
def musical_instruments_equalizer(column1, column2, column3, show_spectro):

    signal_x_axis, signal_y_axis, sample_rate = read_audio(".piano_timpani_piccolo_out.wav")    # read audio file

    yf, points_per_freq = fourier_transform(signal_y_axis, sample_rate)         # Fourier Transfrom

    if (show_spectro):
        plot_spectro(column2,'.piano_timpani_piccolo_out.wav')
    else:
        plotting_graphs(column2,signal_x_axis,signal_y_axis,False)

    sliders_labels = ['Drums', 'Timpani', 'Piccolo']

    yf = equalizer(yf, points_per_freq, 3, sliders_labels, "Music")         #create sliders and modify signal

    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders
    modified_signal_channel = np.int16(modified_signal) # returns two channels 

    write(".Equalized_Music.wav", sample_rate, modified_signal_channel)     # creates the modified song

    if (show_spectro):
        plot_spectro(column3,".Equalized_Music.wav")
    else:
        plotting_graphs(column3,signal_x_axis,modified_signal,False)

    column2.audio('.piano_timpani_piccolo_out.wav', format='audio/wav')    # displaying the audio before editing
    column3.audio(".Equalized_Music.wav", format='audio/wav')              # displaying the audio after  editing

#-------------------------------------- MEDICAL APPLICATION ----------------------------------------------------
def vowels_equalizer(column1, column2, column3, show_spectro):

    signal_x_axis, signal_y_axis, sample_rate = read_audio(".piano_timpani_piccolo_out.wav")    # read audio file
    
    yf, points_per_freq = fourier_transform(signal_y_axis, sample_rate)         # Fourier Transfrom

    if (show_spectro):
        plot_spectro(column2,'.piano_timpani_piccolo_out.wav')
    else:
        plotting_graphs(column2,signal_x_axis,signal_y_axis,False)

    sliders_labels = ['','','','','']

    yf = equalizer(yf, points_per_freq, 5, sliders_labels, "Vowels")         #create sliders and modify signal

    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders
    modified_signal_channel = np.int16(modified_signal) # returns two channels 

    write(".Equalized_Vowels.wav", sample_rate, modified_signal_channel)     # creates the modified song

    if (show_spectro):
        plot_spectro(column3,".Equalized_Vowels.wav")
    else:
        plotting_graphs(column3,signal_x_axis,modified_signal,False)

    column2.audio('.piano_timpani_piccolo_out.wav', format='audio/wav')    # displaying the audio before editing
    column3.audio(".Equalized_Vowels.wav", format='audio/wav')              # displaying the audio after  editing

#-------------------------------------- MEDICAL APPLICATION ----------------------------------------------------
def arrhythima(column1, column2, column3,show_spectro):

    ecg = electrocardiogram()       # Calling the arrhythmia database of a woman
    fs = 360                        # determining f sample
    time = np.arange(ecg.size) / fs # detrmining tima axis

    fourier_y_axis, points_per_freq = fourier_transform(ecg, fs)         # Fourier Transfrom

    sliders_labels = 'Arrhythima'

    fourier_y_axis = equalizer(fourier_y_axis, points_per_freq, 1, sliders_labels, "Arrhythima")

    if (show_spectro):
        pass
    else:
        plotting_graphs(column2,time,ecg,True)

    modified_signal = irfft(fourier_y_axis) 

    if (show_spectro):
        pass
    else:
        plotting_graphs(column3, time, modified_signal, True)

#-------------------------------------- OPTIONAL ----------------------------------------------------
def voice_changer(uploaded_file, column1, column2, column3, show_spectro):

    signal_x_axis, signal_y_axis, sample_rate = read_audio(uploaded_file)

    if (show_spectro):
        plot_spectro(column2, uploaded_file.name)
    else:
        plotting_graphs(column2,signal_x_axis,signal_y_axis,False)

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

    if n_sliders > 1:
        columns=st.columns(n_sliders)
        counter=0
        list_of_sliders_values = []
        while counter < n_sliders:
            with columns[counter]:
                sliders = (st.slider(label=sliders_labels[counter],key=counter, min_value=0, max_value=10,value=1, step=1))
            counter +=1
            list_of_sliders_values.append(sliders)
    else:
        eq_slider = st.slider(label=sliders_labels, min_value=0, max_value=10, value=1, step=1)

    if mode == "Default":
        for index,value in enumerate(list_of_sliders_values):
            yf[int(points_per_freq * 1000 * index)  : int(points_per_freq * 1000 * index + points_per_freq * 1000)] *= value

    elif mode == "Music":
        yf[:int(points_per_freq* 1000)] *= list_of_sliders_values[0]
        yf[int(points_per_freq*1000):int(points_per_freq* 2600)] *= list_of_sliders_values[1]
        yf[int(points_per_freq*2700):] *= list_of_sliders_values[2]

    elif mode == "Vowels":
        pass

    elif mode == "Arrhythima":
        yf[int(points_per_freq*1) : int(points_per_freq* 5)] *= eq_slider

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
def plotting_graphs(column,x_axis,y_axis,flag):
    fig, axs = plt.subplots()
    fig.set_size_inches(6,3)
    plt.plot(x_axis,y_axis)
    if flag == True:
        plt.xlim(45, 55)
        plt.xlabel("Time in s")
        plt.ylabel("ECG in mV")
    column.pyplot(fig)

#-------------------------------------- PLOTING Spectrogram ----------------------------------------------------
def plot_spectro(column,audio_file):
    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    column.pyplot(fig)