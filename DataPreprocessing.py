import numpy as np
import mne
from tqdm import tqdm
from sklearn.decomposition import FastICA
import torch
import os
import scipy.io
from scipy import interpolate
from scipy.signal import butter, filtfilt
import mat73
import warnings
warnings.filterwarnings("ignore")

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
    fs = 500  # sampling frequency
    lowcut = 0.5  # lower cutoff frequency (Hz)
    highcut = 50  # upper cutoff frequency (Hz)
    order = 4  # filter order

    # Calculate the filter coefficients using a Butterworth filter
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(order, [low, high], btype='band')
    # Apply the filter to each channel of the data
    if filter:
        for m in range(eeg.shape[0]):
            eeg[m, :] = filtfilt(b, a, eeg[m, :])
    # Preprocessing with ICA
    if ICA:
        ica = FastICA(n_components=eeg.shape[0], random_state=0)
        eeg = ica.fit_transform(eeg.T).T
    pre_eeg = eeg
    return pre_eeg

#Choose dataset from ['IowaPD', 'UNMPD', 'eeg-motor', 'dataverse']
dataset = 'UNMPD'

# Load the data
if dataset == 'IowaPD':
    data_path = './Datasets/Data and Code/IowaDataset/Organized data'
    #raw_org = h5py.File(os.path.join(organized_data_path, 'IowaData.mat'))
    raw = mat73.loadmat(os.path.join(data_path, 'IowaData.mat'))
    channel_list = list(raw['Channel_location'])
    # Put tensors into a list
    tensors = []
    for i in tqdm(range(2), position=0, leave=False):
        for j in tqdm(range(14), position=0, leave=False):
            try:
                eeg_list = []
                for a in range(63):
                    if a != raw['Channel_location'].index('CPz'):
                        eeg_list.append(raw['EEG'][a][i][j])
                eeg = np.asarray(eeg_list)
                eeg_pre = eeg_preprocess(eeg)
                tensors.append(torch.tensor(eeg_pre, dtype=torch.float32).transpose(0, 1))
            except:
                print(i, j)

    # Pad tensors in the second axis with 0s
    padded_tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True).transpose(1, 2)
    eeg_np = np.concatenate(padded_tensors.numpy(), axis=0)
    # Print the padded tensors
    print(eeg_np.shape)
    data_pre = torch.tensor(eeg_np)
    label = torch.tensor(np.asarray([[1]]*62*14+[[0]]*62*14), dtype=torch.float32)
    channel_list = channel_list[:63]
    channel_list.remove('CPz')
elif dataset == 'UNMPD':
    data_path = './Datasets/Data and Code/UNMDataset'
    raw = mat73.loadmat(os.path.join(data_path, 'EEG_Jim_rest_Unsegmented_WithAllChannels.mat'))
    channel_list = list(raw['Channel_location'])
    # Put tensors into a list
    tensors = []
    for i in tqdm(range(2), position=0, leave=False):
        for j in tqdm(range(27), position=0, leave=False):
            try:
                eeg_list = []
                for a in range(63):
                    if a != raw['Channel_location'].index('Pz'):
                        eeg_list.append(raw['EEG'][a][i][j])
                eeg = np.asarray(eeg_list)
                eeg_pre = eeg_preprocess(eeg)
                tensors.append(torch.tensor(eeg_pre, dtype=torch.float32).transpose(0, 1))
            except:
                print(i, j)

    # Pad tensors in the second axis with 0s
    padded_tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True).transpose(1, 2)
    eeg_np = np.concatenate(padded_tensors.numpy(), axis=0)
    # Print the padded tensors
    print(eeg_np.shape)
    data_pre = torch.tensor(eeg_np)
    label = torch.tensor(np.asarray([[1]]*62*27+[[0]]*62*27), dtype=torch.float32)
    channel_list = channel_list[:63]
    channel_list.remove('Pz')
elif dataset == 'eeg-motor':
    data_path = './Datasets/eeg-motor-movementimagery-dataset-1.0.0'
    channel_file = open(os.path.join(data_path, 'wfdbcal'))
    channel_list = channel_file.read().splitlines()
    channel_file.close()
    channel_list = [n[:3] for n in channel_list]
    filename_txt = open(os.path.join(data_path, 'RECORDS'))
    file_list = filename_txt.read().splitlines()
    filename_txt.close()
    filenames = [f for f in file_list if f[10:12] not in ['01', '02']]
    data_list = []
    for i in tqdm(range(len(filenames))):
        filename = filenames[i]
        # Check if the file is a mat file
        # Load the mat file
        file_path = os.path.join(data_path, filename)
        edf_data = mne.io.read_raw_edf(file_path, verbose=False, preload=True)
        eeg_data = edf_data.get_data()
        eeg_pre = eeg_preprocess(eeg_data)
        data_list.append(torch.tensor(eeg_pre, dtype=torch.float32).transpose(0, 1))
    # Pad tensors in the second axis with 0s
    padded_tensors = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True).transpose(1, 2)
    # Print the padded tensors
    print(padded_tensors.shape)
    data_pre = padded_tensors
    label = torch.tensor([[0],[1],[2],[3]] * 3 * 109, dtype=torch.float32)
elif dataset == 'dataverse':
    data_path = './Datasets/dataverse_files'
    data_list = []
    label = []
    for filename in os.listdir(data_path):
        # Check if the file is a mat file
        if filename.endswith('.mat'):
            # Load the mat file
            file_path = os.path.join(data_path, filename)
            mat_data = scipy.io.loadmat(file_path)
            eeg_data = mat_data['data'].transpose([2,0,1])
            data_list.append(eeg_data)
            label.append(mat_data['label'])
    eeg_np = np.concatenate(data_list, axis=0)
    label = np.concatenate(label, axis=0)
    data_pre = torch.tensor(np.asarray([eeg_preprocess(eeg_np[r], filter=False) for r in tqdm(range(len(eeg_np)))]), dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)

import pickle
save_path = os.path.join(data_path, 'processed')
if not os.path.exists(save_path):
    os.mkdir(save_path)
with open(os.path.join(save_path, 'data.pkl'), 'wb') as file:
    pickle.dump(data_pre.numpy().astype('float32'), file, protocol=4)
    pickle.dump(label.numpy().astype('float32'), file, protocol=4)
    try:
        pickle.dump(np.asarray(channel_list), file, protocol=4)
    except:
        pass