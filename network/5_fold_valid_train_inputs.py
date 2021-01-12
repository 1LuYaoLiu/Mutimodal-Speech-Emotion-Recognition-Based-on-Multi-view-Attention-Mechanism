from train_multimodel_for_inputs import train
import pickle


case = ['two']
case1_element = ['TFM', 'TTT', 'FFF', 'MMM']
case2_element = ['TTM','FFM', 'MFM']


case_dict  = {
    'one':case1_element,
    'two':case2_element,
    # 'three':case3_element,
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
            maxWA, maxUA = train(case=c, element=e, file_num=i)
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

with open('/home/liuluyao/PycharmProjects/NEW_network/model_result/MSP_input/WA.plk', 'wb') as f:
    pickle.dump(WA_dict, f)

with open('/home/liuluyao/PycharmProjects/NEW_network/model_result/MSP_input/UA.plk', 'wb') as f:
    pickle.dump(UA_dict, f)