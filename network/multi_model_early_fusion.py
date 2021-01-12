import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

class MULTIMANET(nn.Module):
    def __init__(self, ahead_text=8, ahidden_text=96,ahead_audio=8, ahidden_audio=96, with_gender=False, case=None, element=None):
        super(MULTIMANET, self).__init__()
        self.case = case
        self.element=element
        self.ahead_text = ahead_text
        self.ahidden_text = ahidden_text
        self.ahead_audio = ahead_audio
        self.ahidden_audio = ahidden_audio

        self.with_gender = with_gender
        self.conv1a = nn.Conv2d(kernel_size=(5, 2), in_channels=1, out_channels=8)
        self.conv1b = nn.Conv2d(kernel_size=(2, 6), in_channels=1, out_channels=8)
        self.conv1c = nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=8)

        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=24, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=64, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=96, padding=1)
        # self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=96, out_channels=128, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn1c = nn.BatchNorm2d(8)

        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(96)
        # self.bn5 = nn.BatchNorm2d(128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=self.ahidden_audio, out_features=4)
        if with_gender == True:
            self.fc2 = nn.Linear(in_features=self.ahidden_audio, out_features=2)
        self.dropout = nn.Dropout(0.5)

        self.dropout_attn_audio = nn.Dropout(0.1)
        # attention based audio
        self.attention_query_audio = nn.ModuleList()
        self.attention_key_audio = nn.ModuleList()
        self.attention_value_audio = nn.ModuleList()

        for i in range(self.ahead_audio):
            self.attention_query_audio.append(nn.Conv2d(in_channels=96, out_channels=self.ahidden_audio, kernel_size=1))
            self.attention_key_audio.append(nn.Conv2d(in_channels=96, out_channels=self.ahidden_audio, kernel_size=1))
            self.attention_value_audio.append(nn.Conv2d(in_channels=96, out_channels=self.ahidden_audio, kernel_size=1))

        self.attention_query_audio2 = nn.ModuleList()
        self.attention_key_audio2 = nn.ModuleList()
        self.attention_value_audio2 = nn.ModuleList()

        for i in range(self.ahead_audio):
            self.attention_query_audio2.append(nn.Conv2d(in_channels=96, out_channels=self.ahidden_audio, kernel_size=1))
            self.attention_key_audio2.append(nn.Conv2d(in_channels=96, out_channels=self.ahidden_audio, kernel_size=1))
            self.attention_value_audio2.append(nn.Conv2d(in_channels=96, out_channels=self.ahidden_audio, kernel_size=1))



        # TEXT process network

        self.wordEmbedding_file = r'/home/liuluyao/IEMOCAP/wordEmbedding.plk'
        with open(self.wordEmbedding_file, 'rb') as f:
            self.wordEmbedding = pickle.load(f)
        # self.wordEmbedding.shape[1]
        self.embedding = torch.nn.Embedding(num_embeddings=self.wordEmbedding.shape[0],
                                            embedding_dim=self.wordEmbedding.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(self.wordEmbedding))

        self.gru = nn.GRU(input_size=200, hidden_size=100, num_layers=1)

        self.fc = nn.Linear(in_features=self.ahidden_text * self.ahead_text, out_features=4)

        self.dropout_attn_text = nn.Dropout(0.1)

        # text self attention

        self.attention_query_text = nn.ModuleList()
        self.attention_key_text = nn.ModuleList()
        self.attention_value_text = nn.ModuleList()

        for i in range(self.ahead_text):
            self.attention_query_text.append(nn.Linear(in_features=100, out_features=self.ahidden_text))
            self.attention_key_text.append(nn.Linear(in_features=100, out_features=self.ahidden_text))
            self.attention_value_text.append(nn.Linear(in_features=100, out_features=self.ahidden_text))


        self.attention_query_text2 = nn.ModuleList()
        self.attention_key_text2 = nn.ModuleList()
        self.attention_value_text2 = nn.ModuleList()

        for i in range(self.ahead_text):
            self.attention_query_text2.append(nn.Linear(in_features=100, out_features=self.ahidden_text))
            self.attention_key_text2.append(nn.Linear(in_features=100, out_features=self.ahidden_text))
            self.attention_value_text2.append(nn.Linear(in_features=100, out_features=self.ahidden_text))



        # normal attention
        self.attn_fc_based_audio = nn.Linear(in_features=96, out_features=100)
        self.attn_fc_based_text = nn.Linear(in_features=100, out_features=96)

        # integral layer  +100+96
        # self.ahidden_text_attn*self.ahead_text_attn+self.ahidden_audio_attn*self.ahead_audio_attn
        in_features_dict = {
            '1':2 * self.ahidden_audio * self.ahead_audio,
            '2':2 * self.ahidden_text * self.ahead_text,
            '3':96,
            '4':100
        }
        self.in_features = 0
        for i in self.element:
            self.in_features += in_features_dict[i]


        self.integral_fc_emotion = nn.Linear(in_features=self.in_features,out_features=4)
        self.integral_fc_emotion_center = nn.Linear(in_features=self.in_features,out_features=4)
        self.integral_fc_sex = nn.Linear(in_features=self.in_features,out_features=2)
        self.integral_fc_sex_center = nn.Linear(in_features=self.in_features,out_features=2)



    def _forward_audio(self, *input):
        # conv1a, kernel_size=(10, 2); input[0]:x_t

        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)

        # conv1b, kernel_size=(2,9); input[1]:x_f

        xb = self.conv1b(input[1])
        xb = self.bn1b(xb)

        xb = F.relu(xb)


        xc = self.conv1b(input[2])
        xc = self.bn1b(xc)

        xc = F.relu(xc)
        # print(xa.shape)
        # print(xb.shape)
        # print(xc.shape)
        xa = F.interpolate(xa, size=(25, 156),mode='bilinear')
        xb = F.interpolate(xb, size=(25, 156), mode='bilinear')
        xc = F.interpolate(xc, size=(25, 156), mode='bilinear')

        x = torch.cat((xa, xb, xc), 1)

        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        # x = self.conv5(x)
        # x = self.bn5(x)
        #
        # x = F.relu(x)

        # #audio_self_attention

        attn = None
        for i in range(self.ahead_audio):
            Q = self.attention_query_audio[i](x)
            K = self.attention_key_audio[i](x)
            V = self.attention_value_audio[i](x)
            attention = F.softmax(torch.mul(Q, K))
            attention = torch.mul(attention, V)

            # attention_img = attention[0, 0, :, :].squeeze().detach().cpu().numpy()
            # img = Image.fromarray(attention_img, 'L')
            # img.save('img/img_'+str(i)+'.png')

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)

        attn2 = None
        for i in range(self.ahead_audio):
            Q = self.attention_query_audio2[i](x)
            K = self.attention_key_audio2[i](x)
            V = self.attention_value_audio2[i](x)
            attention = F.softmax(torch.mul(Q, K))
            attention = torch.mul(attention, V)

            # attention_img = attention[0, 0, :, :].squeeze().detach().cpu().numpy()
            # img = Image.fromarray(attention_img, 'L')
            # img.save('img/img_'+str(i)+'.png')

            if (attn2 is None):
                attn2 = attention
            else:
                attn2 = torch.cat((attn2, attention), 2)



        attn = torch.cat((attn, attn2), 1)

        attn = self.dropout_attn_audio(attn)


        return attn, x

    def _forward_text(self,*input):

        embed_batch = self.embedding(input[0])

        # embed_batch = embed_batch.permute(1,0,2)
        batch_packed = nn.utils.rnn.pack_padded_sequence(embed_batch, input[1], enforce_sorted=False, batch_first=True)

        output, hidden = self.gru(batch_packed)
        # output, hidden = self.gru1(output)
        # audio attention based text
        h = hidden.reshape(1, hidden.shape[1], hidden.shape[2]*hidden.shape[0])[-1]
        # print(x.shape)

        out_padded = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)


        output = out_padded[0].contiguous().view(-1, 100)

        output = self.dropout(output)
        hidden = hidden[0]
        attn = None
        for i in range(self.ahead_text):
            Q = self.attention_query_text[i](output)

            K = self.attention_key_text[i](output)

            V = self.attention_value_text[i](output)

            attention = F.softmax(torch.mul(Q, K))
            attention = torch.mul(attention, V)

            # attention_img = attention[0, 0, :, :].squeeze().detach().cpu().numpy()
            # img = Image.fromarray(attention_img, 'L')
            # img.save('img/img_'+str(i)+'.png')

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 1)

        attn2 = None
        for i in range(self.ahead_text):
            Q = self.attention_query_text2[i](output)

            K = self.attention_key_text2[i](output)

            V = self.attention_value_text2[i](output)

            attention = F.softmax(torch.mul(Q, K))
            attention = torch.mul(attention, V)

            # attention_img = attention[0, 0, :, :].squeeze().detach().cpu().numpy()
            # img = Image.fromarray(attention_img, 'L')
            # img.save('img/img_'+str(i)+'.png')

            if (attn2 is None):
                attn2 = attention
            else:
                attn2 = torch.cat((attn2, attention), 1)

        attn = torch.cat((attn, attn2), 1)

        attn = self.dropout_attn_text(attn)

        return attn, h, out_padded[0].shape[1]


    def forward(self,*input):
        '''
        input[0]:MFCC_t
        input[1]:MFCC_f
        input[2]:trans
        input[3]:trans_length
        '''
        self_attn_audio, x_audio = self._forward_audio(input[0], input[1], input[2])

        self_attn_text, h_text, n_words = self._forward_text(input[3],input[4])

        # self attention of audio
        attention = None
        for i in range(self.ahead_audio):
            n = self_attn_audio.shape[2]/self.ahead_audio
            attn = self.gap(self_attn_audio[:,:,int(i*n):int((i+1)*n),:])
            attn = self.gap(attn)

            if (attention is None):
                attention = attn
            else:
                attention = torch.cat((attention, attn), 1)
        self_attn_audio = attention.reshape(attention.shape[0], attention.shape[1]*attention.shape[2]*attention.shape[3])

        # self attention of text
        self_attn_text = self_attn_text.view(int(self_attn_text.shape[0]/(n_words)), n_words, self_attn_text.shape[1])

        self_attn_text = torch.sum(self_attn_text, dim=1).squeeze()



        # input of attention layers from audio
        x_audio = self.gap(x_audio)
        x_audio = x_audio.reshape(x_audio.shape[0], x_audio.shape[1]*x_audio.shape[2]*x_audio.shape[3])

        # normal attention based audio of text
        a_text = self.attn_fc_based_audio(x_audio)
        a_text =  F.softmax(a_text)
        attn_text_based_audio = torch.mul(a_text, h_text)

        # normal attention based text of audio
        a_audio = self.attn_fc_based_text(h_text)
        a_audio = F.softmax(a_audio)
        attn_audio_based_text = torch.mul(a_audio, x_audio)


        if len(self_attn_text.shape) == 1:
            self_attn_text = self_attn_text.unsqueeze(0)

        attn_dict = {
            '1':self_attn_audio,
            '2':self_attn_text,
            '3':attn_audio_based_text,
            '4':attn_text_based_audio
        }
        # if self.element == '1234':
        #     x = torch.cat((self_attn_text, attn_text_based_audio, self_attn_audio, attn_audio_based_text), 1)
        #     # x = torch.cat(( self_attn_audio, self_attn_text, attn_audio_based_text, attn_text_based_audio), 1)
        if len(self.element) == 1:
            x = attn_dict[self.element]
        elif len(self.element) == 2:
            x = torch.cat((attn_dict[self.element[0]], attn_dict[self.element[1]]), 1)

        elif len(self.element) == 3:
            x = torch.cat((attn_dict[self.element[0]], attn_dict[self.element[1]], attn_dict[self.element[2]]), 1)

        else:
            x = torch.cat((self_attn_text, attn_text_based_audio, self_attn_audio, attn_audio_based_text), 1)



        x = F.relu(x)
        x = self.dropout(x)
        x_emotion = self.integral_fc_emotion(x)
        x_emotion_center = self.integral_fc_emotion_center(x)
        x_sex = self.integral_fc_sex(x)
        x_sex_center = self.integral_fc_sex_center(x)

        #print('scale:', self.fuse_weight_1, self.fuse_weight_2, self.fuse_weight_3, self.fuse_weight_4)
        return x_emotion, x_sex, x_emotion_center, x_sex_center


# class text_LSTM(nn.Module):
#     def __init__(self):

