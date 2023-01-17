import pickle, mne, warnings, copy
import numpy as np
import pandas as pd
from tqdm import trange
from scipy import signal
from itertools import chain
from mne.preprocessing import ICA
from mne.filter import filter_data as bandpass_filter
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

N_C = None
droping_components = 'one'
def SignalPreProcess(eeg_rawdata):
    """
    :param eeg_rawdata: numpy array with the shape of (n_channels, n_samples)
    :return: filtered EEG raw data
    """
    assert eeg_rawdata.shape[0] == 32
    eeg_rawdata = np.array(eeg_rawdata)

    ch_names = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", 
                "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
                "PO4", "O2"]
  
    info = mne.create_info(ch_names = ch_names, ch_types = ['eeg' for _ in range(32)], sfreq = 128, verbose=False)
    raw_data = mne.io.RawArray(eeg_rawdata, info, verbose = False)
    raw_data.load_data(verbose = False).filter(l_freq = 4, h_freq = 48, method = 'fir', verbose = False)
    #raw_data.plot()

    ica = ICA(n_components = N_C, random_state = 97, verbose = False)
    ica.fit(raw_data)
    # https://mne.tools/stable/generated/mne.preprocessing.find_eog_events.html?highlight=find_eog_#mne.preprocessing.find_eog_events
    eog_indices, eog_scores = ica.find_bads_eog(raw_data.copy(), ch_name = 'Fp1', verbose = None)
    a = abs(eog_scores).tolist()
    if(droping_components == 'one'):
        ica.exclude = [a.index(max(a))]
        
    else: # find two maximum scores
        a_2 = a.copy()
        a.sort(reverse = True)
        exclude_index = []
        for i in range(0, 2):
            for j in range(0, len(a_2)):
                if(a[i]==a_2[j]):
                    exclude_index.append(j)
        ica.exclude = exclude_index
    ica.apply(raw_data, verbose = False)
    # common average reference
    raw_data.set_eeg_reference('average', ch_type = 'eeg')#, projection = True)
    filted_eeg_rawdata = np.array(raw_data.get_data())
    return filted_eeg_rawdata


def signal_pro(input_data):
    # signal processing
    print('Data preprocessing:')
    for i in trange(input_data.shape[0]):
        input_data[i] = SignalPreProcess(input_data[i].copy())
    return input_data


def get_class_labels(labels, class_type):
    # encoding
    emotion = np.ones(40)
    if(class_type=='valence'):
        for i in range(0, 40):
            if labels[i][0]>=5:
                emotion[i] = 0
            else:
                emotion[i] = 1
    elif(class_type=='arousal'):
        for i in range(40):
            if labels[i][1]>=5:
                emotion[i] = 0
            else:
                emotion[i] = 1
    else:
        for i in range(40):
            if(labels[i][0]>=5 and labels[i][1] >=5): # HVHA
                emotion[i] = 0
            elif(labels[i][0]>=5 and labels[i][1]<5): #HVLA
                emotion[i] = 1
            elif(labels[i][0]<5 and labels[i][1]>=5): #LVHA
                emotion[i] = 2
            else: #LVLA
                emotion[i] = 3
    return emotion


def get_data(dataset_path, subject_no):
    # read the data
    deap_dataset = pickle.load(open(dataset_path + subject_no + '.dat', 'rb'), encoding='latin1')
    # separate data and labels 
    data = np.array(deap_dataset['data']) # for current data
    labels = np.array(deap_dataset['labels']) # for current labels
    # remove 3sec pre baseline
    data  = data[0:40, 0:32, 384:8064]
    return signal_pro(data), get_class_labels(labels, 'four_class')