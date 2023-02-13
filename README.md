### [ DSP_Task2_Team7 ](https://github.com/Ayasameh1/DSP-Equalizer/blob/main/Task_Info.md)

# DSP-Equalizer
#### Python app using Streamlit for adjusting frequencies in audio wav files and csv.
 
## Description
This web app is based on fourier transform, by transferring the normal signal into fourier domain, processing it, and finally applying inverse fourier to view changes.

**This equalizer allows the user to:**
- Add .wav audio file and change each frequency power in the default mode.
- Add a music file and control its Timpani, drums, and piccolo sound.
- Add a sound file and control vowels on it, either to hide the vowel or to strengthen it, it's applicable on 3 vowels & 2 constants, ['ʃ','ʊ','a','r','b'].
- Turn a normal voice into deeper or smoother voice.
- Show the spectrogram of the audio signal.
- Play and pause the dynamic plot of the audio signal before and after modifying the frequency. 

 ## Overview 
 This app has 5 modes :
 
 **1. Default**
 - Divides the sginal frequency range into 10 sliders by which the user can control a specific range. 

**2. Voice changer**
 - Turns a normal human voice into deeper or smoother voice.

 **3. Vowels equalizer**
 - Allows the user to hide a specific vowel from a sentence.

 **4. Music equalizer**
 - Controlls three instrument power by multiplying the amplitude by a factor from 0 (to hide the instrument) to 10.

 **5. Medical** 
 - Control arrhythmia component in ECG signal.


 ## Dependencies
 Python 3.10
 #### used libraries
 - streamlit
 - wave
 - librosa
 - scipy
 - numpy
 - pandas
 - plotly.express
 - matplotlib.pyplot
 - altair
 
  ## Preview
- #### Home page
![main · Streamlit - Personal - Microsoft​ Edge 11_18_2022 9_58_25 PM](https://user-images.githubusercontent.com/93640020/202792104-c6202c4c-6195-47e9-9a18-e539ecc0812c.png)

- #### Dynamic plotting on default mode



https://user-images.githubusercontent.com/93640020/202792417-d7380bb3-51a6-43c4-a2e3-c9928a356b75.mp4



- #### Music mode



https://user-images.githubusercontent.com/93640020/202783054-fb4e0852-5cb2-4984-9319-6e1fca931c5f.mp4



- #### Vowels mode



https://user-images.githubusercontent.com/93640020/202783019-51ce2037-cbcc-401e-bc86-214927fb9e7c.mp4



- #### Arrhythmia mode 



https://user-images.githubusercontent.com/93640020/202783083-b9483b4c-6421-4743-b46b-2381a9e73038.mp4



- #### Voice changer



https://user-images.githubusercontent.com/93640020/202783473-3d7d91f6-1122-420e-b838-a4398048fb7b.mp4

- Course Name : Digital Signal Processing .

## Submitted to:

- Dr. Tamer Basha & Eng. Abdullah Darwish

All rights reserved © 2022 to Team 7 - Systems & Biomedical Engineering, Cairo University (Class 2024)
 
