from train_multimodel import train
import numpy as np
import pandas as pd
import pickle


case = [1,2,3,4]
case1_element = ['1','2','3','4']
case2_element = ['12', '13', '14', '23', '24', '34']
case3_element = ['123', '124', '134', '234']
case4_element = ['1234']
attn_dict = {
    1:'audio_self_attention',
    2:'text_self_attention',
    3:'audio_attention_based_text',
    4:'text_attention_based_audio'
}
case_dict  = {
    1:case1_element,
    2:case2_element,
    3:case3_element,
    4:case4_element
}
WA_dict = {}
UA_dict = {}
for c in case:
    for e in case_dict[c]:


        print('*************************************************case_'+str(c)+'_training_'+e+'*************************************')
        file_num = [1,2,3,4,5]
        WA = []
        UA = []
        for i in file_num:
            print('******************************************************train_'+str(i)+'***********************************')
            print(i, c, e)
            maxWA, maxUA = train(file_num=i, case=c, element=e)
            WA.append(maxWA)
            UA.append(maxUA)
        acc_dict = {
            'WA':WA,
            'UA':UA
        }
        WA_dict['case'+str(c)+'_'+e] = WA
        UA_dict['case' + str(c) + '_' + e] = UA
        print('case_'+str(c)+'_training_'+e+'_WA:',WA)
        print('case_' + str(c) + '_training_' + e + '_UA:', UA)

with open('/home/liuluyao/PycharmProjects/NEW_network/model_result/IEMOCAP_attention/WA.plk', 'wb') as f:
    pickle.dump(WA_dict, f)

with open('/home/liuluyao/PycharmProjects/NEW_network/model_result/IEMOCAP_attention/UA.plk', 'wb') as f:
    pickle.dump(UA_dict, f)