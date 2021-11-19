import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as font_manager
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (roc_curve ,precision_recall_curve ,average_precision_score)
'''
import sys
import os
sys.path.append(os.getcwd() + '\\my_version\\version2')
os.chdir('./my_version/version2')
'''
from model import sequence_model
(
    full_question,
    full_division,
    tokenized_question,
    labeled_key,
    division_list
) = pickle.load(open(f"./pickle/test_data.pkl", "rb"))
pd.Series([d for subd in full_division for d in subd]).value_counts()
sample_count = dict.fromkeys(division_list ,0)
sample_question,sample_division,sample_token,sample_key = [],[],[],[]
bre_count = 0
bre = False
for (q,d,t,l) in zip(full_question,full_division,tokenized_question,labeled_key):
    for div in d:
        sample_count[div] += 1
    if max(sample_count.values()) > 10:
        bre = True
    if min(sample_count.values()) >= 10:
        bre = False
    if bre :
        for div in d :
            sample_count[div] -= 1
        bre = False
        continue

    sample_question.append(q)
    sample_division.append(d)
    sample_token.append(t)
    sample_key.append(l)
    bre_count += 1
    if bre_count == 260:
        break
sum(sample_count.values())

with open("./pickle/250_data.pkl", "wb") as file:
    pickle.dump((sample_question, sample_division, sample_token, sample_key, division_list), file)


pd.Series([d for subd in sample_division for d in subd]).value_counts()
[(i,q) for i,q in enumerate(sample_question) if '圖' in q]
pd.DataFrame({"Question" : sample_question,
              "Label" : sample_division}).to_csv('division_prediction250.csv',encoding='utf-8-sig')



model = sequence_model()
model.load_model('1f1')

true_label, probability = model.test(batch_size = 32 ,save = False , if_return= True)

precision = dict()
recall = dict()
average_precision = dict()
for i in range(24):
    precision[i], recall[i], _ = precision_recall_curve(true_label[:, i],
                                                        probability[:, i])
    average_precision[i] = average_precision_score(true_label[:, i], probability[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(true_label.ravel(),
    probability.ravel())
average_precision["micro"] = average_precision_score(true_label, probability,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))


def plot_curve( label, division_list, prob, curve='PR', beta=1):
    assert curve in ['ROC', 'PR']
    e = 1e-20
    colors = cm.rainbow(np.linspace(0, 1, 6))
    ##AUC,AP and threshold by division
    for i, c in zip([1, 6, 7, 12, 17, 19], colors):
        if curve == 'ROC':
            fpr, tpr, threshold_list_1 = roc_curve(label[:, i], prob[:, i])
            index1 = np.argmax((tpr[1:] - fpr[1:]))
            plt.plot(fpr, tpr, label=division_list[i])
            plt.scatter(fpr[index1], tpr[index1])
        else:
            sub_precision, sub_recall, thresholds_list_2 = precision_recall_curve(label[:, i], prob[:, i])

            sub_f1_score = 2 / (1 / ((sub_precision + e) / (beta + 1)) + beta / ((sub_recall + e) / (beta + 1)))
            index2 = np.argmax(sub_f1_score)
            plt.plot(sub_recall[::-1], sub_precision[::-1], label=division_list[i], c=c)
            plt.scatter(sub_recall[index2], sub_precision[index2], marker="o", c=c)

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(curve + 'curve')
    font = font_manager.FontProperties(family="SimSun",
                                       weight='bold',
                                       style='normal', size=10)
    plt.legend(prop=font)
    plt.savefig(curve + 'curve')