from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

import librosa


from flask import Flask,request,render_template

app = Flask(__name__)

dataset_path = './Respiratory_Sound_Database/audio_and_txt_files/'
diagnosis = pd.read_csv('./Respiratory_Sound_Database/patient_diagnosis.csv',header=None)

wav_files = []
diagnosis_dict = {}

labels = []
audio_features = []
health_conditions = {"COPD":0, "Healthy":1, "URTI":2, "Bronchiectasis":3, "Pneumonia":4, "Bronchiolitis":5, "Asthma":6, "LRTI":7}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_wav_files_list")
def get_wav_files_list():
    global wav_files
    for wav_file in listdir(dataset_path):
        if isfile(join(dataset_path, wav_file)) and wav_file.endswith('.wav'):
            wav_files.append(wav_file)
    print("Available Audio(wav) Files {}".format(len(wav_files)))
    return {"wav_files": wav_files}

@app.route("/patients_in_dataset")
def patients_in_dataset():
    global diagnosis_dict
    for index, row in diagnosis.iterrows():
        diagnosis_dict[row[0]] = row[1]
    print("Number of Patients {0}".format(len(diagnosis_dict)))
    print(diagnosis_dict)
    return {"diagnosis_dict": diagnosis_dict}

@app.route("/load_audio_files")
def load_audio_files():
    global labels
    global audio_features
    count = 0
    for wav_file in wav_files:
        sound, sample_rate = librosa.load(dataset_path + wav_file)
        stft = np.abs(librosa.stft(sound))
        mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate), axis=1)
        mel = np.mean(librosa.feature.melspectrogram(y=sound, sr=sample_rate), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate), axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate), axis=1)
        concat = np.concatenate((mfccs, chroma, mel, contrast, tonnetz))
        labels.append(health_conditions[diagnosis_dict[int(wav_file[:3])]])
        audio_features.append(concat)
        count = count + 1
        if health_conditions[diagnosis_dict[int(wav_file[:3])]] != 7 and health_conditions[
            diagnosis_dict[int(wav_file[:3])]] != 6:
            print("Loaded {0}/{1}".format(count, len(wav_files)))
    labels = np.array(labels)
    audio_features = np.array(audio_features)
    return {"message": str(labels)+" audio files belongs to "+str(diagnosis_dict)+" patients"}
app.run(debug=True)