import numpy as np
from tqdm import trange
from utils import time_domain_features, frequency_domain_features
from utils import Rational_Differential_Asymmetry, dwt_features


def get_optimal_channels_no(eeg_channels, optimal_channels):
    channel_no = []
    for ithchannel in optimal_channels:
        for i in range(0, len(eeg_channels)):
            if(ithchannel==eeg_channels[i]):
                channel_no.append(i)
                break
    return channel_no


def get_features(data, optimal_channels):
    eeg_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                    'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 
                    'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    channel_no = get_optimal_channels_no(eeg_channels, optimal_channels) # only taking selected channels
    feature_matrix = []
    print("Multi-Domain feature extraction:")
    for ith_video in trange(0, 40):
        features = []
        for ith_channel in channel_no:
            input_data = data[ith_video][ith_channel]
            time_feat = time_domain_features(input_data)
            freq_feat = frequency_domain_features(input_data)
            dtw_feat  = dwt_features(input_data)
            features = features + time_feat + freq_feat + dtw_feat
        # Add Rational_Differential_Asymmetry in freq_freq
        # Note here the input data shape will be like 2D (channels * data_points)
        DASM_RASM = Rational_Differential_Asymmetry(data[ith_video], optimal_channels)
        features = features + DASM_RASM
        # flatten the features i.e. transform it from 2D to 1D
        feature_matrix.append(features)
    return np.array(feature_matrix)