import numpy as np
import torch
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from Trainer import test_Model
import os
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def linechart_plot_save(x, path):
    plt.plot(list(range(len(x))), x)
    plt.savefig(path)
    plt.show()

def select_features_chi2(X_train, y_train, dim):
    fs = SelectKBest(score_func=chi2, k=dim)
    #        X_train = MinMaxScaler().fit_transform(X_train)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    return X_train_fs


def select_features_mul(X_train, y_train, dim):
    fs = SelectKBest(score_func=mutual_info_classif, k=dim)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    return X_train_fs

def aver_record(record, length=1024):
    maxlen = length

    if len(record) < maxlen:
        record = np.asarray(record + [0] * (maxlen - len(record))).astype('float32')
    else:
        chunk_list = []
        for j in range(int(len(record) / maxlen) + 1):
            if len(record[j * maxlen:(j + 1) * maxlen]) == maxlen:
                chunk_list.append(np.asarray(record[j * maxlen:(j + 1) * maxlen]))
            else:
                chunk_list.append(np.asarray(
                    record[j * maxlen:(j + 1) * maxlen] + [0] * (
                            maxlen - len(record[j * maxlen:(j + 1) * maxlen]))).astype('float32'))
        record = np.asarray(sum(chunk_list) / len(chunk_list))
    return record

if __name__ == '__main__':
    #Choose dataset from ['IowaPD', 'UNMPD']
    feature_selection = True
    # Set device and parameters
    if torch.cuda.is_available():
        torch.cuda.set_device(1)

    data_path1 = './Datasets/Data and Code/IowaDataset/Organized data/processed/data.pkl'
    data_path2 = './Datasets/Data and Code/UNMDataset/processed/data.pkl'
    num_classes = 2


    with open(data_path1, 'rb') as file:
        x1 = pickle.load(file).astype('float32')
        y1 = pickle.load(file).astype('float32')
        channel_list1 = pickle.load(file)

    with open(data_path2, 'rb') as file:
        x2 = pickle.load(file).astype('float32')
        y2 = pickle.load(file).astype('float32')
        channel_list2 = pickle.load(file)

    if feature_selection:
        k_dim1 = 60000
        x1 = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1))
        x1 = select_features_chi2(x1, y1, k_dim1)

#        x2 = x2[:, :120000]
        k_dim2 = 120000
        x2 = (x2 - np.min(x2)) / (np.max(x2) - np.min(x2))
        x2 = select_features_chi2(x2, y2, k_dim2)

    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)

    model_path1 = os.path.join(data_path1[:-18], 'Model')
    model_path2 = os.path.join(data_path2[:-18], 'Model')

    model_test1 = torch.load(os.path.join(model_path1, 'best_model.pth'))
    model_test2 = torch.load(os.path.join(model_path2, 'best_model.pth'))

    _, predictions1_1 = test_Model(x1[:int(x1.shape[0] / 2)], y1[:int(x1.shape[0] / 2)], model_test1)
    _, predictions1_2 = test_Model(x1[-int(x1.shape[0] / 2):], y1[-int(x1.shape[0] / 2):], model_test1)
    predictions1 = list(predictions1_1) + list(predictions1_2)

    predictions2 = []
    part = 27
    for p in range(part):
        _, predictions2_p = test_Model(x2[p*int(x2.shape[0]/part):(p+1)*int(x2.shape[0]/part)], y2[p*int(x2.shape[0]/part):(p+1)*int(x2.shape[0]/part)], model_test2)
        predictions2 += list(predictions2_p)

    predictions1 = np.asarray(predictions1)
    labels1 = y1.reshape((-1))
    correct1 = np.asarray((predictions1 - labels1) == 0).astype('int')
    channel_dic1 = {}
    for i in range(62):
        acc = sum([correct1[a] for a in range(correct1.shape[0]) if a % 62 == i])/(correct1.shape[0]/62)
        channel_dic1[channel_list1[i]] = acc
    sorted_channel1 = dict(sorted(channel_dic1.items(), key=lambda x: x[0]))

    predictions2 = np.asarray(predictions2)
    labels2 = y2.reshape((-1))
    correct2 = np.asarray((predictions2 - labels2) == 0).astype('int')
    channel_dic2 = {}
    for i in range(62):
        acc = sum([correct2[a] for a in range(correct2.shape[0]) if a % 62 == i]) / (correct2.shape[0] / 62)
        channel_dic2[channel_list2[i]] = acc
    sorted_channel2 = dict(sorted(channel_dic2.items(), key=lambda x: x[0]))

    channel_dic = {}
    for k in sorted_channel1.keys():
        channel_dic[k] = (sorted_channel1[k] + sorted_channel2[k])/2

    channel_dic = dict(sorted(channel_dic.items(), key=lambda x: x[1]))

    plt.figure(figsize=(15, 20))
    plt.barh(list(channel_dic.keys()), list(channel_dic.values()), color='green')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim((0.96, 1.001))
    plt.title('Channel Performance Comparison', fontsize=20)
    plt.savefig('./Charts/chart7.png')