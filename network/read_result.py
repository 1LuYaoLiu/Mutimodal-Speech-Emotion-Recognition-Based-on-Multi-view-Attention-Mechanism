import pickle
import os
import glob
import numpy as np

attn_UA_file = r'/home/liuluyao/PycharmProjects/NEW_network/model_result/IEMOCAP_attention/UA.plk'

attn_WA_file = r'/home/liuluyao/PycharmProjects/NEW_network/model_result/IEMOCAP_attention/WA.plk'

attn_matrix_file = glob.glob(r'/home/liuluyao/PycharmProjects/NEW_network/model_result/IEMOCAP/based_model/1_1_matrix.plk')

with open(attn_UA_file, 'rb') as uaf:
    attn_UA = pickle.load(uaf)

print(attn_UA)
average_attn_UA = {}
for key in attn_UA.keys():
    average_attn_UA[key] = np.mean(np.array(attn_UA[key]))

with open(attn_WA_file, 'rb') as waf:
    attn_WA = pickle.load(waf)
print(attn_WA)

average_attn_WA = {}
for key in attn_WA.keys():
    average_attn_WA[key] = np.mean(np.array(attn_WA[key]))
with open('./IEMOCAP_UA.plk', 'wb') as f:
    pickle.dump(average_attn_UA, f)
with open('./IEMOCAP_WA.plk', 'wb') as f:
    pickle.dump(average_attn_WA, f)
print('UA:', average_attn_UA)
print('WA:', average_attn_WA)

for m in attn_matrix_file:
    print(os.path.basename(m))



    with open(m, 'rb') as f:
        matrix = pickle.load(f)
        print(matrix)
        per_matrix = []
        num_class = np.sum(matrix, axis=1).reshape(1,-1).getA().tolist()[0]
        print(num_class)
        for i in range(len(matrix)):
            per_matrix.append([x/num_class[i] for x in matrix[i]])
        percentage = []
        acc_class = []
        for i in range(len(per_matrix)):
            percentage.append(per_matrix[i][0].getA().tolist()[0])
        for i in range(len(percentage)):
            acc_class.append(percentage[i][i])
        # print(os.path.basename(m), ': ', np.matrix(np.array(percentage)))
        # print(os.path.basename(m), '_acc_class:', acc_class)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
sns.set()
f, ax = plt.subplots()
a = sns.heatmap(percentage, annot=True, ax=ax, cmap="Blues", fmt='.2f',
                xticklabels=['Neural', 'Sad', 'Angry', 'Happy'], yticklabels=['Neural', 'Sad', 'Angry', 'Happy'],
                annot_kws={'family':'Times New Roman','weight':'normal','size': 14})

# ax.set_title('confusion matrix') #标题
ax.set_xlabel('Predict') #x轴
ax.set_ylabel('True') #y轴
plt.show()




