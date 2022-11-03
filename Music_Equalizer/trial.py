import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.signal import find_peaks
import streamlit as st
import plotly.graph_objects as go


SAMPLE_RATE = 44100  # Hertz
DURATION = 5  # Seconds
fig = go.Figure()

# uploaded_file = st.file_uploader(label="Uploading Signal", type = ['csv',".wav"])

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

def FT(signal,Fs,duration):

    DURATION = duration
    SAMPLE_RATE = int(Fs)
    mixed_tone = signal
    normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

    # plt.plot(normalized_tone[:1000])
    # plt.show()

    # Remember SAMPLE_RATE = 44100 Hz is our playback rate
    write("myUploadedSignal.wav", SAMPLE_RATE, normalized_tone)


    # Number of samples in normalized_tone
    N = int(SAMPLE_RATE * DURATION)

    yf = rfft(normalized_tone)
    # powers = yf * np.conj(yf) / N
    # indices = powers > 1
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    peaks_indeces = find_peaks(yf)
    print(peaks_indeces)
    plt.plot(xf, np.abs(yf))
    plt.plot(peaks_indeces, color='black' , marker="o" ,linestyle="")
    plt.show()

    # The maximum frequency is half the sample rate
    points_per_freq = len(xf) / (SAMPLE_RATE / 2)

    # Our target frequency is 4000 Hz
    target_idx = int(points_per_freq * 10)
    target_idx_2 = int(points_per_freq * 4000)
    yf[target_idx - 1 : target_idx_2 + 2] = 0

    # plt.plot(xf, np.abs(yf))
    # plt.show()


    new_sig = irfft(yf)

    # plt.plot(new_sig[:1000])
    # plt.show()

    norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))

    write("cleanUploaded04.wav", SAMPLE_RATE, norm_new_sig)

from scipy.io.wavfile import read,write
from IPython.display import Audio

Fs, data = read('mixkit-retro-game-emergency-alarm-1000.wav')
data = data[:,0]
duration = data.shape[0]/Fs
FT(data,Fs,duration)
