import numpy as np
import torch
import random
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from Models import EEGNet, EEGNet_1C, EEGNet_1C_2
from Trainer import train, test_Model
import os
import pandas as pd
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
    dataset = 'IowaPD'
    partial = False
    turncate = True
    feature_selection = True
    # Set device and parameters
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
    lr = 0.001
    n_epochs = 50
    batch_size = 128
    verbose = 1

    # Load the data
    if dataset == 'IowaPD':
        data_path = './Datasets/Data and Code/IowaDataset/Organized data/processed/data.pkl'
        num_classes = 2
    elif dataset == 'UNMPD':
        data_path = './Datasets/Data and Code/UNMDataset/processed/data.pkl'
        num_classes = 2
    else:
        data_path = './'
        num_classes = 2


    with open(data_path, 'rb') as file:
        x = pickle.load(file).astype('float32')
        y = pickle.load(file).astype('float32')
        channel_list = pickle.load(file)

    if turncate and (dataset == 'IowaPD' or dataset == 'UNMPD'):
        if feature_selection:
            if dataset == 'IowaPD':
                k_dim = 60000
            if dataset == 'IowaPD':
                k_dim = 120000
            else:
                k_dim = 60000
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            x = select_features_chi2(x, y, k_dim)
        else:
            x = x[:, :170000]

    if partial:
        random.seed(33)
        rand = random.sample(list(range(len(x))), 1200)
        x = x[rand]
        y = y[rand]



    if dataset == 'IowaPD':
        x = np.expand_dims(x, axis=1)
        model = EEGNet_1C(num_classes, x.shape[1], x.shape[2]).cuda()
    elif dataset == 'UNMPD':
        x = np.expand_dims(x, axis=1)
        model = EEGNet_1C(num_classes, x.shape[1], x.shape[2]).cuda()
    else:
        model = EEGNet(num_classes, x.shape[1], x.shape[2]).cuda()


    one_hot = preprocessing.OneHotEncoder(sparse=False)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)
    y_train = one_hot.fit_transform(y_train.reshape(-1, 1))
    y_test = one_hot.fit_transform(y_test.reshape(-1, 1))

    model_path = os.path.join(data_path[:-18], 'Model')
    result_path = os.path.join(data_path[:-18], 'Result')

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    trained_model, loss_list, valid_loss_list, val_acc_list = train(model, x_train, y_train, x_test, y_test, n_epochs, batch_size, lr,
                                                      root=model_path)

    metrics_list = test_Model(x_test, y_test, trained_model)

    writer = pd.ExcelWriter(os.path.join(result_path, 'Result_CNN.xls'))
    result = pd.DataFrame({'Metrics': ['Acc', 'Auc', 'Precision', 'F1', 'Recall'], 'Result': metrics_list})
    loss = pd.DataFrame({'loss': loss_list, 'valid_loss': valid_loss_list, 'val_acc': val_acc_list})
    loss.to_excel(os.path.join(result_path, 'Loss_CNN.xls'))

    result.to_excel(excel_writer=writer)
    writer.save()
    writer.close()

    model_test = torch.load(os.path.join(model_path, 'best_model.pth'))
    _, predictions_1 = test_Model(np.expand_dims(x[:int(x.shape[0]/4)], axis=1), y[:int(x.shape[0]/4)], model_test)
    _, predictions_2 = test_Model(np.expand_dims(x[int(x.shape[0]/4):2*int(x.shape[0]/4)], axis=1), y[int(x.shape[0]/4):2*int(x.shape[0]/4)], model_test)
    _, predictions_3 = test_Model(np.expand_dims(x[2*int(x.shape[0]/4):3*int(x.shape[0]/4)], axis=1), y[2*int(x.shape[0]/4):3*int(x.shape[0]/4)], model_test)
    _, predictions_4 = test_Model(np.expand_dims(x[3*int(x.shape[0]/4):], axis=1), y[3*int(x.shape[0]/4):], model_test)
    predictions = list(predictions_1) + list(predictions_2) + list(predictions_3) + list(predictions_4)
    collab_preds = [np.round(np.sum(predictions[62*i:62*(i+1)])/62) for i in range(int(x.shape[0]/62))]
    label = [1.0]*int(x.shape[0]/124) + [0.0]*int(x.shape[0]/124)
    Acc = metrics.accuracy_score(label, collab_preds)
    Precision = metrics.precision_score(y_true=label, y_pred=collab_preds, zero_division=0)
    Auc = metrics.roc_auc_score(label, collab_preds)
    F1 = metrics.f1_score(label, collab_preds)
    Recall = metrics.recall_score(label, collab_preds)
    print('Acc: {} , Auc: {} , Pre: {} , F1: {} , Recall: {} '.format(Acc, Auc, Precision, F1, Recall))