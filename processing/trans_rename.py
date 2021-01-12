import pandas as pd
import numpy as np
import os

from nltk.tokenize import word_tokenize
import nltk

LABEL = {
    'neu': '01',  #: 'neutral',
    'fru': '02',  #: 'calm',
    'hap': '03',  #: 'happy',
    'sad': '04',  #: 'sad',
    'ang': '05',  #: 'angry',
    'fea': '06',  #: 'fearful',
    'exc': '07',  #: 'disgust',
    'sur': '08',  #: 'surprised'
    'xxx': '09',  #: 'other'
}

trans = pd.read_csv(r'H:/IEMOCAP/processed_tran.csv',names=['name', 'trans'])
label = pd.read_csv(r'H:/IEMOCAP/label.csv',names=['name', 'label'])



for i in range(len(trans)):
    k = trans.iloc[i,0].split('_')
    if label.iloc[i, 1] not in LABEL.keys():
        label.iloc[i, 1] = 'xxx'
    if len(k) == 3:
        trans.iloc[i, 0] = k[0] + '-' + k[1] + '-' + LABEL[label.iloc[i, 1]] +'-'+ k[-1][-4] +'-'+ k[2]
    elif len(k) == 4:
        trans.iloc[i,0] = k[0] + '-' + k[1] + '-' + LABEL[label.iloc[i, 1]] +'-'+ k[-1][-4] +'-'+ k[2] + '_' + k[3]

trans.to_csv(r'H:/IEMOCAP/trans.csv', index=0)