import numpy as np
from tqdm import tqdm
from sklearn.decomposition import FastICA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import torch
import os
from scipy import interpolate
from scipy.signal import butter, filtfilt
import mat73
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def select_features_chi2(X_train, y_train, dim):
    fs = SelectKBest(score_func=chi2, k=dim)
    #        X_train = MinMaxScaler().fit_transform(X_train)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    return X_train_fs

def remove_outliers(eeg_data, z_thresh=3):
    # Calculate Z-score for each sample in each channel
    z_scores = np.abs((eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True))
    # Replace outliers with NaN values
    eeg_data[z_scores > z_thresh] = np.nan
    # Interpolate missing values using linear interpolation
    for i in range(eeg_data.shape[0]):
        eeg_data[i, :] = interpolate.interp1d(np.arange(eeg_data.shape[1])[~np.isnan(eeg_data[i, :])],
                                              eeg_data[i, ~np.isnan(eeg_data[i, :])],
                                              kind='linear',
                                              fill_value='extrapolate')(np.arange(eeg_data.shape[1]))
    return eeg_data


def eeg_preprocess(eeg, filter=True, ICA=True):
    # Apply standardization to each channel of the data
    eeg = remove_outliers(eeg)
    eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / np.std(eeg, axis=1, keepdims=True)
    # Define the filter parameters
    fs = 250  # sampling frequency
    lowcut = 0.5  # lower cutoff frequency (Hz)
    highcut = 50  # upper cutoff frequency (Hz)
    order = 4  # filter order

    # Calculate the filter coefficients using a Butterworth filter
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(order, [low, high], btype='band')
    # Apply the filter to each channel of the data

    eeg_filter = eeg
    if filter:
        for m in range(eeg_filter.shape[0]):
            eeg_filter[m, :] = filtfilt(b, a, eeg_filter[m, :])
    # Preprocessing with ICA

    if ICA:
        ica = FastICA(n_components=eeg_filter.shape[0], random_state=0)
        eeg_filter = ica.fit_transform(eeg_filter.T).T
    return eeg, eeg_filter

#Choose dataset from ['IowaPD', 'UNMPD']
dataset = 'UNMPD'

# Load the data
data_path = './Datasets/Data and Code/IowaDataset/Organized data'
#raw_org = h5py.File(os.path.join(organized_data_path, 'IowaData.mat'))
raw = mat73.loadmat(os.path.join(data_path, 'IowaData.mat'))
channel_list = list(raw['Channel_location'])
# Put tensors into a list
tensor_raw = []
tensor_std = []
tensor_filter = []
for i in tqdm(range(2), position=0, leave=False):
    for j in tqdm(range(1), position=0, leave=False):
        try:
            eeg_list = []
            for a in range(63):
                if a != raw['Channel_location'].index('CPz'):
                    eeg_list.append(raw['EEG'][a][i][j])
            eeg = np.asarray(eeg_list).astype('float32')
            tensor_raw.append(eeg)
            eeg_std, eeg_filter = eeg_preprocess(eeg)
            tensor_std.append(eeg_std)
            tensor_filter.append(eeg_filter)
        except:
            print(i, j)

fig1 = plt.figure(figsize=(20, 10))
f1ax1 = fig1.add_subplot(2,1,1)
f1ax1.set_title('EEG recording of PD patient (Fp1)', fontsize=30)
f1ax2 = fig1.add_subplot(2,1,2)
f1ax2.set_title('EEG recording of control subject (Fp1)', fontsize=30)

f1ax1.plot(list(range(tensor_raw[0][0][:2000].shape[0])), tensor_raw[0][0][:2000], color='k')
f1ax2.plot(list(range(tensor_raw[1][0][:2000].shape[0])), tensor_raw[1][0][:2000], color='k')

fig2 = plt.figure(figsize=(20, 10))
f2ax1 = fig2.add_subplot(2,1,1)
f2ax1.set_title('Referenced and Standardized EEG recording of PD patient (Fp1)', fontsize=30)
f2ax2 = fig2.add_subplot(2,1,2)
f2ax2.set_title('Referenced and Standardized EEG recording of control subject (Fp1)', fontsize=30)

f2ax1.plot(list(range(tensor_std[0][0][:2000].shape[0])), tensor_std[0][0][:2000], color='k')
f2ax2.plot(list(range(tensor_std[1][0][:2000].shape[0])), tensor_std[1][0][:2000], color='k')

fig3 = plt.figure(figsize=(20, 10))
f3ax1 = fig3.add_subplot(2,1,1)
f3ax1.set_title('Preprocessed EEG recording of PD patient (Fp1)', fontsize=30)
f3ax2 = fig3.add_subplot(2,1,2)
f3ax2.set_title('Preprocessed EEG recording of control subject (Fp1)', fontsize=30)

f3ax1.plot(list(range(tensor_filter[0][0][:2000].shape[0])), tensor_filter[0][0][:2000], color='k')
f3ax2.plot(list(range(tensor_filter[1][0][:2000].shape[0])), tensor_filter[1][0][:2000], color='k')

k_dim = 60000
y = torch.tensor([1, 0])
tensor_list = [torch.tensor(tensor[0, :], dtype=torch.float32).reshape(-1, 1) for tensor in tensor_filter]
padded_tensors = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True).transpose(1, 2)
x = np.concatenate(padded_tensors.numpy(), axis=0)
x = (x - np.min(x)) / (np.max(x) - np.min(x))
x = select_features_chi2(x, y, k_dim)

fig4 = plt.figure(figsize=(20, 10))
f4ax1 = fig4.add_subplot(2,1,1)
f4ax1.set_title('EEG recording of PD patient (Fp1)', fontsize=30)
f4ax2 = fig4.add_subplot(2,1,2)
f4ax2.set_title('EEG recording of control subject (Fp1)', fontsize=30)

f4ax1.plot(list(range(x[0][:2000].shape[0])), x[0][:2000], color='k')
f4ax2.plot(list(range(x[1][:2000].shape[0])), x[1][:2000], color='k')

fig1.savefig('./Charts/chart1.png')
fig2.savefig('./Charts/chart2.png')
fig3.savefig('./Charts/chart3.png')
fig4.savefig('./Charts/chart4.png')

data_UNM = './Datasets/Data and Code/UNMDataset/Result/Loss_CNN.xls'
data_Iowa = './Datasets/Data and Code/IowaDataset/Organized data/Result/Loss_CNN.xls'

UNM_xls = pd.read_excel(data_UNM)
Iowa_xls = pd.read_excel(data_Iowa)

legend_list = ['Train_loss', 'Val_loss']

fig5 = plt.figure(figsize=(15, 6))
f5ax1 = fig5.add_subplot(1,2,1)
f5ax1.set_title('Training and validation loss', fontsize=20)
f5ax2 = fig5.add_subplot(1,2,2)
f5ax2.set_title('Validation accuracy', fontsize=20)

f5ax1.plot(list(range(len(UNM_xls))), list(UNM_xls['loss']), linewidth=2.5)
f5ax1.plot(list(range(len(UNM_xls))), list(UNM_xls['valid_loss']), linewidth=2.5)
f5ax1.legend(legend_list)

f5ax2.plot(list(range(len(UNM_xls))), list(UNM_xls['val_acc']), color='green', linewidth=2.5)

fig5.savefig('./Charts/chart5.png')

fig6 = plt.figure(figsize=(15, 6))
f6ax1 = fig6.add_subplot(1,2,1)
f6ax1.set_title('Training and validation loss', fontsize=20)
f6ax2 = fig6.add_subplot(1,2,2)
f6ax2.set_title('Validation accuracy', fontsize=20)

f6ax1.plot(list(range(len(Iowa_xls))), list(Iowa_xls['loss']), linewidth=2.5)
f6ax1.plot(list(range(len(Iowa_xls))), list(Iowa_xls['valid_loss']), linewidth=2.5)
f6ax1.legend(legend_list)

f6ax2.plot(list(range(len(Iowa_xls))), list(Iowa_xls['val_acc']), color='green', linewidth=2.5)

fig6.savefig('./Charts/chart6.png')