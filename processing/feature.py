import librosa as rosa
import numpy as np
from python_speech_features import sigproc

class FeatureExtractor(object):
    def __init__(self,rate):
        self.rate = rate


    def get_feature(self,feature_to_use, x):
        x_feature = None
        # accepted_features_to_use = ('mfcc_2018', 'mfcc', 'mfcc_f', 'mfcc_t',  'melspectrogram_f', 'melspectrogram_t', 'spectrogram', 'prosody')
        # if feature_to_use not in accepted_features_to_use:
        #     raise NotImplementedError("{} not in {}!".format(feature_to_use, accepted_features_to_use))
        # else:

        if feature_to_use in ('mfcc_f'):
            # 128ms, 32ms
            x_feature = self.get_mfcc(x, win_length=1024, hop_length=512, n_mfcc=39)

        if feature_to_use in ('mfcc_t'):
            # 40ms, 20ms
            x_feature = self.get_mfcc(x,win_length=512, hop_length=256, n_mfcc=13)


        if feature_to_use in ('mfcc'):
            x_feature = self.get_mfcc(x, win_length=768, hop_length=384, n_mfcc=26)



        # if feature_to_use in ('mfcc_2018'):
        #     mfcc = rosa.feature.mfcc(np.array(x), win_length=400, hop_length=160, n_mfcc=13)
        #     delta = rosa.feature.delta(mfcc)
        #     delta_delta = rosa.feature.delta(delta)
        #     x_feature = np.concatenate((mfcc, delta, delta_delta), 0)

        if feature_to_use in ('melspectrogram_f'):
            x_feature = self.get_melspectrogram(x, n_fft=2048, hop_length=1024, n_mels=196)
        if feature_to_use in ('melspectrogram_t'):
            x_feature = self.get_melspectrogram(x, n_fft=640, hop_length=320, n_mels=64)
        if feature_to_use in ('melspectrogram_m'):
            x_feature = self.get_melspectrogram(x, n_fft=1024, hop_length=512, n_mels=128)


        return x_feature

    def get_mfcc(self,x, win_length=1024, hop_length=512, n_mfcc=13):
        def _get_mfcc(x):
            mfcc_data = rosa.feature.mfcc(x,sr=self.rate, n_mfcc=n_mfcc, win_length=win_length, hop_length=hop_length)
            return mfcc_data
        x_feature = np.apply_along_axis(_get_mfcc,1,x)
        return x_feature

    def get_SFF_spec(self,x):
        def _get_SFF_spec(x):
            pass
        x_feature = np.apply_along_axis(_get_SFF_spec, 1, x)
        return

    def get_melspectrogram(self, X, n_fft=2048, hop_length=512, n_mels=128):
        def _get_melspectrogram(x):
            # [np.newaxis,:]
            mel = rosa.feature.melspectrogram(y=x, sr=self.rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            # delta = rosa.feature.delta(mel)
            # delta_delta = rosa.feature.delta(delta)
            # out = np.concatenate((mel, delta, delta_delta))
            return mel

        X_features = np.apply_along_axis(_get_melspectrogram,1, X)
        return X_features

    def get_spectrogram(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features




# if __name__ == '__main__':
#     wav = r'/home/liuluyao/IEMOCAP/wav/Ses01F-impro01-01-01-F000.wav'
#     y, sr = rosa.load(wav, sr=None)
#     feature_use = 'mfcc'
#     feature_extractor = FeatureExtractor(sr)
#     mfcc_ = feature_extractor.get_feature(feature_use,y)
