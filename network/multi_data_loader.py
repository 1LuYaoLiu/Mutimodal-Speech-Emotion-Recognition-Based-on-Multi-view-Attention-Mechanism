import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

dict = {
    0: torch.Tensor([0]),
    1: torch.Tensor([1]),
    2: torch.Tensor([2]),
    3: torch.Tensor([3]),
}


class DataSet(Dataset):
    def __init__(self, X_f,X_t,X_m, trans,trans_len, Y, sex):
        self.X_f = X_f
        self.X_t = X_t
        self.X_m = X_m
        self.trans = trans
        self.trans_len = trans_len
        self.Y = Y
        self.sex = sex

    def __getitem__(self, index):
        x_f = self.X_f[index]

        x_f = torch.from_numpy(x_f.astype(np.float32))
        x_f = x_f.float()

        x_t = self.X_t[index]
        # x = torch.from_numpy(x).unsqueeze(0)
        x_t = torch.from_numpy(x_t)
        x_t = x_t.float()

        x_m = self.X_m[index]
        # x = torch.from_numpy(x).unsqueeze(0)
        x_m = torch.from_numpy(x_m)
        x_m = x_m.float()

        trans = self.trans[index]
        # x = torch.from_numpy(x).unsqueeze(0)
        trans = torch.from_numpy(trans)
        trans = trans.long()

        trans_len = self.trans_len[index]


        y = self.Y[index]
        y = dict[y]
        y = y.long()

        sex = self.sex[index]
        sex = dict[sex]
        sex = sex.long()


        return x_f, x_t,x_m, trans, trans_len, y, sex

    def __len__(self):
        return len(self.X_f)

# if __name__=='__main__':
#     import pickle
#     file = r'../processing/features_mfcc_all.pkl'
#     with open(file, 'rb') as f:
#         features = pickle.load(f)
#
#     val_X_f = features['val_X_f']
#     val_X_t = features['val_X_t']
#     val_y = features['val_y']
#     val_sex = features['val_sex']
#
#     val_data = DataSet(val_X_f, val_X_t, val_y, val_sex)
#     val_loader = DataLoader(val_data, batch_size=10, shuffle=True)
#     for i ,data in enumerate(val_loader):
#         x_f, x_t, y, sex = data
#         print(y)
#         print(sex)
#         print(val_X_f.shape)
#         print(val_X_t.shape)
#         break

