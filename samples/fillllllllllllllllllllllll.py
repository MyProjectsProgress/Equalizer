from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
import wave
import numpy as np
spf = wave.open('short.wav', "r")
print( spf )
data = spf.readframes(-1)
data = np.frombuffer(data, "int32")
rate = spf.getframerate()
fft_out = fft(data)
fft_out=np.abs(fft_out)[:len(data)//2]
xf = fftfreq(len(data),(1/rate))[:len(data)//2]
plt.plot(xf,fft_out)
plt.show()
