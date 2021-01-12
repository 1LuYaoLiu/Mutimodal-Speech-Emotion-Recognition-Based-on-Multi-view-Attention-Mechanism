import librosa as rosa
import numpy as np
import glob
from tqdm import tqdm
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from processing.silence_delete import delete_silence
from processing.feature import FeatureExtractor



def process(path, t=4, RATE=16000, without_silence=True, win_length=2048,hop_length=512,to_save_feature=True):
    path = path.rstrip('/')
    wav_files = glob.glob(path + '/*.wav')
    train_len = []
    val_len = []


    # get trans_int-type text feature
    with open(r'/home/liuluyao/MSP_IMPROV/MSP_trans_ids.plk', 'rb') as f:
        trans_feature = pickle.load(f)



    LABEL_DICT1 = {
        'N': 'neutral',
        # '02': 'frustration',
        # '03': 'happy',
        'S': 'sad',
        'A': 'angry',
        # '06': 'fearful',
        'H': 'happy'  # excitement->happy
        # '08': 'surprised'
    }

    label_num = {
        'neutral': 0,
        'happy': 0,
        'sad': 0,
        'angry': 0,
    }

    n = len(wav_files)
    print(n)

    cluster1 = list(np.random.choice(range(n), int(n * 0.2), replace=False))
    rest = list(set(range(n)) - set(cluster1))
    cluster2 = list(np.random.choice(rest, int(n*0.2), replace=False))
    rest = list(set(rest)-set(cluster2))
    cluster3 = list(np.random.choice(rest, int(n*0.2), replace=False))
    rest = list(set(rest) - set(cluster3))
    cluster4 = list(np.random.choice(rest, int(n * 0.2), replace=False))
    rest = list(set(rest) - set(cluster4))
    cluster5 = rest
    # print(len(cluster1))
    # print(len(cluster2))
    # print(len(cluster3))
    # print(len(cluster4))
    # print(len(cluster5))
    indices_dict = {
        'cluster1':cluster1,
        'cluster2':cluster2,
        'cluster3':cluster3,
        'cluster4':cluster4,
        'cluster5':cluster5,
    }
    for leave in range(5):
        train_files = []
        valid_files = []
        meta_dict = {}
        val_dict = {}
        features_file_name = './{}_features_{}_MFCC.pkl'.format('leave'+str(leave+1), 'MSP')
        leave_cluster = 'cluster'+str(leave+1)
        valid_indices = []
        train_indices = []
        for k in indices_dict.keys():
            if k==leave_cluster:
                valid_indices.extend(indices_dict[k])
            else:
                train_indices.extend(indices_dict[k])

        # print(len(valid_indices))
        # print(len(train_indices))
        for i in train_indices:
            train_files.append(wav_files[i])
        for i in valid_indices:
            valid_files.append(wav_files[i])

        print("constructing meta dictionary for {}...".format(path))


        for i, wav_file in enumerate(tqdm(train_files)):

            label = os.path.basename(wav_file).split('-')[2][-1]
            label = LABEL_DICT1[label]
            label_num[label] +=1


            wav_data, _ = rosa.load(wav_file, sr=RATE)
            x1 = []
            y1 = []
            index = 0

            if without_silence==True:
                wav_data, _, __ = delete_silence(wav_data, frame_length=win_length, hop_length=hop_length)

            if (t * RATE >= len(wav_data)):
                wav_data = list(wav_data)
                wav_data.extend(np.zeros(int(t * RATE - len(wav_data))))
                # # Signal normalization
                # signal = wav_data
                # signal = np.double(signal)
                #
                # signal = signal / (2.0 ** 15)
                # DC = signal.mean()
                # MAX = (np.abs(signal)).max()
                # signal = (signal - DC) / MAX
                # wav_data = signal
                x1 = wav_data

                y1.append(label)
            else:
                wav_data = list(wav_data[:int(t * RATE)])
                # # Signal normalization
                # signal = wav_data
                # signal = np.double(signal)
                #
                # signal = signal / (2.0 ** 15)
                # DC = signal.mean()
                # MAX = (np.abs(signal)).max()
                # signal = (signal - DC) / MAX
                # wav_data = signal
                x1 = wav_data
                y1.append(label)

            if os.path.basename(wav_file)[:-4] not in trans_feature.keys():
                continue

            if os.path.basename(wav_file)[:-4] not in trans_feature.keys():
                print(os.path.basename(wav_file)[:-4])
                continue
            trans = trans_feature[os.path.basename(wav_file)[:-4]]
            # if len(trans) <100:
            #     trans.extend(np.zeros(int(100 - len(trans))))
            # else:
            #     trans = trans[:100]
            meta_dict[i] = {
                'X': x1,
                'trans':trans,
                'trans_len':len(trans),
                'y': y1,
                'path': wav_file,
                'sex':os.path.basename(wav_file).split('-')[3][0]
            }
        print('label_num:', label_num)
        print("building X, y...")
        train_X = []
        train_trans = []
        train_trans_len = []
        train_y = []
        train_sex = []
        y_dict = {
            'neutral':0,
            'sad':1,
            'angry':2,
            'happy':3
        }
        sex_dict = {
            'F':0,
            'M':1
        }

        for k in meta_dict:
            if meta_dict[k]['y'][0] == 'neutral':
                for i in range(2):
                    train_X.append(meta_dict[k]['X'])
                    trans = meta_dict[k]['trans']
                    train_trans.append(trans)
                    train_trans_len.append(meta_dict[k]['trans_len'])

                    train_y.append(y_dict[meta_dict[k]['y'][0]])
                    train_sex.append(sex_dict[meta_dict[k]['sex'][0]])
            else:
                train_X.append(meta_dict[k]['X'])
                trans = meta_dict[k]['trans']
                train_trans.append(trans)
                train_trans_len.append(meta_dict[k]['trans_len'])

                train_y.append(y_dict[meta_dict[k]['y'][0]])
                train_sex.append(sex_dict[meta_dict[k]['sex'][0]])

        train_X = np.row_stack(train_X)
        train_trans = np.array(train_trans)
        train_trans_len = np.array(train_trans_len)

        train_y = np.array(train_y)
        train_sex = np.array(train_sex)
        assert len(train_X) == len(train_y) == len(train_sex)==len(train_trans)==len(train_trans_len), "X length, y length and sex length must match! X shape: {}, y length: {}, sex length: {}".format(
            train_X.shape, train_y.shape, train_sex.shape)
        print(train_X.shape, train_y.shape, train_sex.shape, train_trans.shape, train_trans_len.shape)
        print(train_trans_len)

        print("valid data process....")
        for i, wav_file in enumerate(tqdm(valid_files)):

            label = os.path.basename(wav_file).split('-')[2][-1]
            label = LABEL_DICT1[label]
            wav_data, _ = rosa.load(wav_file, sr=RATE)
            x1 = []
            y1 = []

            if without_silence == True:
                wav_data, _, __ = delete_silence(wav_data, frame_length=win_length, hop_length=hop_length)

            if (t * RATE >= len(wav_data)):
                wav_data = list(wav_data)
                wav_data.extend(np.zeros(int(t * RATE - len(wav_data))))
                # # Signal normalization
                # signal = wav_data
                # signal = np.double(signal)
                #
                # signal = signal / (2.0 ** 15)
                # DC = signal.mean()
                # MAX = (np.abs(signal)).max()
                # signal = (signal - DC) / MAX
                # wav_data = signal
                x1 = wav_data

                y1.append(label)
            else:

                wav_data = list(wav_data[:int(t * RATE)])
                # # Signal normalization
                # signal = wav_data
                # signal = np.double(signal)
                #
                # signal = signal / (2.0 ** 15)
                # DC = signal.mean()
                # MAX = (np.abs(signal)).max()
                # signal = (signal - DC) / MAX
                # wav_data = signal
                x1 = wav_data
                y1.append(label)
            if os.path.basename(wav_file)[:-4] not in trans_feature.keys():
                print(os.path.basename(wav_file)[:-4])
                continue
            trans = trans_feature[os.path.basename(wav_file)[:-4]]
            # if len(trans) < 100:
            #     trans.extend(np.zeros(int(100 - len(trans))))
            # else:
            #     trans = trans[:100]
            val_dict[i] = {
                'X': x1,
                'trans': trans,
                'trans_len': len(trans),
                'y': y1,
                'path': wav_file,
                'sex': os.path.basename(wav_file).split('-')[3][0]
            }

        val_X = []
        val_trans = []
        val_trans_len = []

        val_y = []
        val_sex = []
        for k in val_dict:
            val_X.append(val_dict[k]['X'])
            trans = val_dict[k]['trans']
            val_trans.append(trans)
            val_trans_len.append(val_dict[k]['trans_len'])

            val_y.append(y_dict[val_dict[k]['y'][0]])
            val_sex.append(sex_dict[val_dict[k]['sex'][0]])

        val_X = np.row_stack(val_X)
        val_trans = np.array(val_trans)
        val_trans_len = np.array(val_trans_len)

        val_y = np.array(val_y)
        val_sex = np.array(val_sex)
        assert len(val_X) == len(val_y)==len(val_sex)==len(val_trans)==len(val_trans_len), "X length ,y length and sex length must match! X shape: {}, y length: {}, sex length: {}".format(
        val_X.shape, val_y.shape, val_sex.shape)
        print(val_X.shape, val_y.shape, val_trans.shape, val_trans_len.shape)

        print(train_X.shape)
        # print(train_y)
        # print(train_sex)
        # print(val_X.shape)
        # print(val_y)
        # print(val_sex)
        train_X_f = process_features(train_X,rate=16000, feature_to_use='mfcc_f')
        train_X_t = process_features(train_X, rate=16000, feature_to_use='mfcc_t')
        train_X_m = process_features(train_X, rate=16000, feature_to_use='mfcc')

        val_X_f = process_features(val_X,rate=16000,feature_to_use='mfcc_f')
        val_X_t = process_features(val_X, rate=16000,feature_to_use='mfcc_t')
        val_X_m = process_features(val_X, rate=16000, feature_to_use='mfcc')

        print('train_X_f:', train_X_f.shape)
        print('train_X_t:', train_X_t.shape)
        print('train_X_m:', train_X_m.shape)
        print('val_X_f:', val_X_f.shape)
        print('val_X_t:', val_X_t.shape)
        print('val_X_m:', val_X_m.shape)

        train_len.append(train_X_t.shape[0])
        val_len.append(val_X_t.shape[0])

        if to_save_feature==True:
            features = {
                'train_X_f':train_X_f,
                'train_X_t':train_X_t,
                'train_X_m':train_X_m,
                'train_trans':train_trans,
                'train_trans_len':train_trans_len,

                'train_y':train_y,
                'train_sex':train_sex,
                'val_X_f':val_X_f,
                'val_X_t':val_X_t,
                'val_X_m':val_X_m,
                'val_trans':val_trans,
                'val_trans_len':val_trans_len,
                'val_y':val_y,
                'val_sex':val_sex
            }
            with open(features_file_name, 'wb') as f:
                pickle.dump(features, f)


    print(train_len)
    print(val_len)


def process_features(X, rate, feature_to_use, u=255):
    feature_extractor = FeatureExtractor(rate)
    X_feature = feature_extractor.get_feature(feature_to_use=feature_to_use, x=X)
    return X_feature


process(r'/home/liuluyao/MSP_IMPROV/wav')
# if __name__ == '__main__':
#     file = r'/home/liuluyao/IEMOCAP/wav/Ses01F-impro01-02-M-M007.wav'
#     y, sr = rosa.load(file, sr=None)
#
#     wav_data, _, __ = delete_silence(y, frame_length=2048, hop_length=512)
#
#     if (4 * sr >= len(wav_data)):
#         wav_data = list(wav_data)
#         wav_data.extend(np.zeros(int(4 * sr - len(wav_data))))
#     else:
#         wav_data = list(wav_data[:int(4*sr)])
#
#     print(np.array(wav_data).reshape(1, len(wav_data)).shape)
#
#     melspec = process_features(np.array(wav_data).reshape(1, len(wav_data)), rate=sr, feature_to_use='melspectrogram')
#     print(melspec.shape)