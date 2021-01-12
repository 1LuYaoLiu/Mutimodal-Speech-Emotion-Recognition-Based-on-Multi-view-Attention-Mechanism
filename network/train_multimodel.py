import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import librosa
from center_loss import CenterLoss
from tqdm import tqdm
import pickle
import logging
import time
import random
import warnings
warnings.filterwarnings('ignore')

from multi_data_loader import DataSet
from multi_model_early_fusion import MULTIMANET

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



# setup_seed(111111)
# setup_seed(123456)
# setup_seed(0)
# setup_seed(999999)
def train(file_num, case, element):
    setup_seed(987654)

    # logger setting
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件

    log_name = 'train.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)


    # parameters setting
    alpha = 0.7
    beta = 0.3
    center_rate = 0.1
    learning_rate = 0.001
    lr_cent = 0.15
    Epoch = 10
    BatchSize = 32
    # MODEL_NAME='MULITMANET_with_gender'

    #MODEL_PATH = './model_result/IEMOCAP.pth'.format(str(case),element, file_num)

    # load features file
    print('load features data ...')
    logging.info('load features data...')
    file = r'./processing/leave{}_features_IEMOCAP_MFCC.pkl'.format(file_num)
    with open(file, 'rb') as f:
        features = pickle.load(f)

    val_X_f = features['val_X_f']
    val_X_t = features['val_X_t']
    val_X_m = features['val_X_m']
    val_trans = features['val_trans']
    val_trans_len = features['val_trans_len']

    val_y = features['val_y']
    val_sex = features['val_sex']

    train_X_f = features['train_X_f']
    train_X_t = features['train_X_t']
    train_X_m = features['train_X_m']
    train_trans = features['train_trans']
    train_trans_len = features['train_trans_len']


    train_y = features['train_y']
    train_sex = features['train_sex']
    print(train_X_m.shape)
    print(train_X_t.shape)
    print(train_X_f.shape)


    '''training processing'''
    print('start training...')
    logging.info('start training....')
    # load data
    #  train_trans, train_trans_len,
    train_data = DataSet(train_X_f, train_X_t, train_X_m,train_trans, train_trans_len, train_y, train_sex)
    train_loader = DataLoader(train_data, batch_size=BatchSize, shuffle=True)

    # load model
    # ahead_text = 7, ahidden_text = 96
    model = MULTIMANET(ahead_text=7, ahidden_text=96,ahead_audio=4, ahidden_audio=96, with_gender=True,case=case, element=element)

    if torch.cuda.is_available():
        model = model.cuda()

    # criterion
    criterion = nn.CrossEntropyLoss()
    center_loss = CenterLoss(num_classes=4, feat_dim=4, use_gpu=False)
    center_loss_sex = CenterLoss(num_classes=2, feat_dim=2, use_gpu=False)

    params = list(model.parameters()) +list(center_loss.parameters())+list(center_loss_sex.parameters())
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=1e-6)

    # result saving
    maxWA = 0
    maxUA = 0
    totalrunningtime = 0

    for i in range(Epoch):
        start_time = time.clock()
        tq = tqdm(len(train_y))

        model.train()
        print_loss=0
        j = 0
        for _,data in enumerate(train_loader):
            x_f, x_t, x_m, trans, trans_len, y, sex= data
            if torch.cuda.is_available():
                x_f = x_f.cuda()
                x_t = x_t.cuda()
                x_m = x_m.cuda()
                trans = trans.cuda()
                trans_len = trans_len.cuda()
                y = y.cuda()
                sex = sex.cuda()

            out_emotion, out_gender, out_emotion_center, out_gender_center = model(x_t.unsqueeze(1), x_f.unsqueeze(1),
                                                                                   x_m.unsqueeze(1), trans, trans_len)

            loss_emotion = criterion(out_emotion, y.squeeze(1))
            center_loss_emotion = center_loss(out_emotion_center, y.squeeze(1))

            loss_gender = criterion(out_gender, sex.squeeze(1))
            center_loss_gender = center_loss_sex(out_gender_center, sex.squeeze(1))

            loss = alpha*(loss_emotion + center_rate*center_loss_emotion)+beta * (loss_gender + center_rate * center_loss_gender)

            print_loss += loss.data.item()*BatchSize
            optimizer.zero_grad()
            loss.backward()
            for param in center_loss.parameters():
                param.grad.data *= (lr_cent/(center_rate*learning_rate))
            optimizer.step()
            tq.update(BatchSize)
        tq.close()
        print('epoch: {}, loss: {:.4}'.format(i, print_loss/len(train_y)))
        logging.info('epoch: {}, loss: {:.4}'.format(i, print_loss))
        if i>0 and i%10 == 0:
            learning_rate = learning_rate/10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        '''validation process'''
        end_time = time.clock()
        totalrunningtime += end_time-start_time
        print('total_running_time:', totalrunningtime)
        model.eval()
        UA = [0, 0, 0, 0]
        num_correct = 0
        class_total = [0, 0, 0, 0]
        matrix = np.mat(np.zeros((4,4)), dtype=int)

        for i in range(len(val_y)):
            x_f = torch.from_numpy(val_X_f[i]).float()
            x_t = torch.from_numpy(val_X_t[i]).float()
            x_m = torch.from_numpy(val_X_m[i]).float()
            trans = torch.from_numpy(np.array(val_trans[i])).long()
            trans_len = val_trans_len[i]

            y = torch.from_numpy(np.array(val_y[i])).long()
            sex = torch.from_numpy(np.array(val_sex[i])).long()
            if torch.cuda.is_available():
                x_f = x_f.cuda()
                x_t = x_t.cuda()
                x_m = x_m.cuda()
                trans = trans.cuda()
                y = y.cuda()
                sex = sex.cuda()


            out_emotion, out_gender, out_emotion_center, out_gender_center= model(x_t.unsqueeze(0).unsqueeze(0), x_f.unsqueeze(0).unsqueeze(0), x_m.unsqueeze(0).unsqueeze(0), trans.unsqueeze(0), torch.tensor([trans_len]))
            pred_emotion = torch.max(out_emotion, 1)[1]
            pred_gender = torch.max(out_gender, 1)[1]

            if pred_emotion[0] == y.item():
                num_correct +=1
            matrix[int(y.item()), int(pred_emotion[0])] +=1

        for i in range(4):
            for j in range(4):
                class_total[i] += matrix[i,j]
            UA[i] = round(matrix[i, i] / class_total[i], 3)
        WA = num_correct/ len(val_y)
        if (maxWA<WA):
            maxWA=WA
            #torch.save(model.state_dict(), MODEL_PATH)
            matrix_file = './model_result/IEMOCAP/based_model/{}_{}_matrix.plk'.format(element, file_num)

            with open(matrix_file, 'wb') as f:
                pickle.dump(matrix, f)
        if (maxUA < sum(UA) / 4):
            maxUA = sum(UA) / 4

        print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
        logging.info('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
        print(matrix)

        logging.info(matrix)
    return maxWA, maxUA




