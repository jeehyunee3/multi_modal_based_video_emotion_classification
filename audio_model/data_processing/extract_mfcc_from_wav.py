# Load packages
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sklearn


# 1. Set paths
origin_path='./origin_path'
save_path='./save_path'


# 2.load file
audio_files = tqdm(list(Path(origin_path).glob("**/*.wav")))

# 3. Get and save MFCC Images from wav file
### Target image size
FIG_SIZE = (13,8)

for audioFile in audio_files:
  ### Get signal and sampling rate information from file
  plt.clf()
  name = audioFile.stem
  audio, sr = librosa.load(str(audioFile), sr=16000)

  ### Get MFCC
  MFCCs = librosa.feature.mfcc(audio, sr, n_mfcc=10, n_fft=1028, hop_length=256)
  MFCCs = sklearn.preprocessing.scale(MFCCs, axis=1)

  pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
  padded_mfcc = pad2d(MFCCs, 40)
  plt.clf()
  plt.figure(figsize=FIG_SIZE)
  librosa.display.specshow(padded_mfcc, sr=sr, hop_length=hop_length)

  ### Save image
  plt.savefig(save_path+"\\"+name+'.png', bbox_inches='tight')
  plt.show()


print("finish save")