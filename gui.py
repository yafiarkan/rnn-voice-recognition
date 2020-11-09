#import tkinter as tk
from tkinter import messagebox
from tkinter import *
import pyaudio
import wave
import pandas as pd
import os
import librosa
import numpy as np
import librosa
import keras
import scipy.io.wavfile as wav
import python_speech_features
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from sklearn import metrics


koef_mfcc = 13
no_path = []
res = []
x = ""
audio_duration = 0
strd = []
endd = []
inh = []
rstrt = False
rstrt2 = 0
path = ""
nm = 0
lstm = False

def extract_data_test(start_duration, end_duration):
    #load wav
    audio, sfreq = librosa.load(path, offset=start_duration, duration = end_duration-start_duration , sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sfreq, n_mfcc=koef_mfcc)
    return mfcc.T

def myClick():
    global nm
    nm = e.get()

def open_lstm():
    global lstm
    lstm = True
    global koef_mfcc
    koef_mfcc = 7
    j=Tk()
    j.title('Command Detection')
    global e
    e=Entry(j,width=50)
    e.pack()
    e.insert(0,"enter")
    myButton = Button(j,text="Generate",command=myClick)
    myButton.pack()
    button_h = Button(j, text="Run", width=25, command=testCallback)
    button_h.pack()

def open_cnn():
    global lstm
    lstm = False
    global koef_mfcc
    koef_mfcc = 13
    j=Tk()
    j.title('Command Detection')
    global e
    e=Entry(j,width=50)
    e.pack()
    e.insert(0,"enter")
    myButton = Button(j,text="Generate",command=myClick)
    myButton.pack()
    button_h = Button(j, text="Run", width=25, command=testCallback)
    button_h.pack()

def create_model():
    if lstm == False:
        model = keras.models.Sequential([
            keras.layers.Conv1D(filters=20, kernel_size=4, strides=1, padding="same", input_shape=[None, koef_mfcc]),
            keras.layers.Conv1D(filters=15, kernel_size=4, strides=1, padding="same"),
            keras.layers.Conv1D(filters=15, kernel_size=4, strides=1, padding="same"),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dense(3, activation='softmax')
            ])
    else:
        model = keras.models.Sequential([
            keras.layers.LSTM(15, return_sequences=True, input_shape=[None, koef_mfcc]),
            keras.layers.LSTM(15),
            keras.layers.Dense(3, activation='softmax')])
    return model

def testCallback():
    global path
    global rstrt
    global x
    global res
    rstrt = False
    num = nm
    path = "cut/"+str(num)+"_uji 1.wav"
    audio, sfreq = librosa.load(path, sr=16000)
    audio_duration =  librosa.get_duration(audio, sr=sfreq)
    word_classifier = create_model()
    if lstm == False:
        word_classifier.load_weights("conv1d_model_koef_"+str(koef_mfcc)+".h5")
    else:
        word_classifier.load_weights("lstm_model_koef_"+str(koef_mfcc)+".h5")
        
    word = ["NYALA", "MATI", "LAINNYA"]
    for i in range(1, math.ceil(audio_duration)+1):
        end_duration = i
        start_duration = i-1
    
        end_duration += 0.15
        start_duration-=0.01
    
        if start_duration<0 : start_duration = 0
    
        if end_duration>audio_duration : end_duration = audio_duration
    

        if end_duration>audio_duration : end_duration=audio_duration
    
        audio = extract_data_test(start_duration, end_duration)
        shape = audio.shape
        audio_p = audio.reshape(1, shape[0], shape[1])
        hasil = word_classifier.predict(audio_p)
        inhasil = np.argmax(hasil)
        strd.append(start_duration)
        endd.append(end_duration)
        inh.append(word[inhasil])

    if not rstrt:
        res.clear()
        x = ""
        for j in range(len(inh)):
            res.append("detik :"+str(strd[j])+"-"+str(endd[j])+" ="+ str(inh[j])+"\n")
        for k in range(len(res)):
            x += res[k]
    rstrt = True
    messagebox.showinfo(title = 'Hasil', message = x)

def restart():
    os.system('python "gui.py"')
    os._exit()
    
j=Tk()
j.title('Command Detection CNN')
buttonh = Button(j, text="LSTM", width=25, command=open_lstm)
buttonh.pack()
buttonk = Button(j, text="CNN", width=25, command=open_cnn)
buttonk.pack()
buttonl = Button(j, text="Restart", width=25, command=restart)
buttonl.pack()
j.mainloop()





