'''
This code is developed by Shyam Marjit.
If you use this code in your research, please kindly cite the following papers

@INPROCEEDINGS{shyam2021eeg,
    title={Enhancing {EEG-}Based Emotion Recognition using MultiDomain Features and Genetic Algorithm based Feature Selection},
    author={Marjit, Shyam and Talukdar, Upasana and Hazarika, Shyamanta M},
    booktitle={9th International Conference on Pattern Recognition and Machine Intelligence},
    year={2021},
    pages={-},
    organization={Springer},
    doi={}
}
'''
import numpy as np
import pandas as pd
from tqdm import trange
from GAMLP import GA_MLP
from feature_selection import GAFS
from data_preprocessing import get_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from MultiDomain_Feature_Extraction import get_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from utils import decoding

# get 10-fold features along with four-class labels
def kfold(x, y):
    # drop constant features if any
    x = x.loc[:,x.apply(pd.Series.nunique) != 1]
    feature_names = x.columns
    x = x.to_numpy()
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x, y)
    test_data, train_data, train_label, test_label = [], [], [], []
    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = pd.DataFrame(data = X_train, columns = feature_names)
        X_test = pd.DataFrame(data = X_test, columns = feature_names)
        train_data.append(X_train)
        test_data.append(X_test)
        train_label.append(y_train)
        test_label.append(y_test)
    return train_data, train_label, test_data, test_label

def binary_to_four(val, ar):
    labels = []
    assert len(val)==len(ar)
    for i in range(0, len(val)):
        if(val[i]==1 and ar[i]==1): # HVHA
            labels.append(0)
        elif(val[i]==1 and ar[i]==0): #HVLA
            labels.append(1)
        elif(val[i]==0 and ar[i]==1): #LVHA
            labels.append(2)
        else: #LVLA
            labels.append(3)
    return labels

def four_to_binary(input_label):
	labels_val, labels_ar = [], []
	for i in input_label:
	    if(i==0): # HVHA
	        labels_val.append(1)
	        labels_ar.append(1)
	    elif(i==1): #HVLA
	        labels_val.append(1)
	        labels_ar.append(0)
	    elif(i==2): #LVHA
	        labels_val.append(0)
	        labels_ar.append(1)
	    else: #LVLA
	        labels_val.append(0)
	        labels_ar.append(0)
	return labels_val, labels_ar

def ind_to_feat(all_feats, ind):
    features = []
    for i in range(0, len(ind)):
        if(ind[i]==1):
            features.append(all_feats[i])
    return features

def performance(y_pred, y_test):
    acc = accuracy_score(y_pred, y_test)*100
    prec = precision_score(y_test, y_pred)*100
    recall = recall_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred)*100
    return acc, prec, recall, f1

def multi_class_performance(y_pred, y_test):
    acc = accuracy_score(y_pred, y_test)*100
    prec = precision_score(y_test, y_pred, average = 'weighted')*100
    recall = recall_score(y_test, y_pred, average = 'weighted')*100
    f1 = f1_score(y_test, y_pred, average = 'weighted')*100
    return acc, prec, recall, f1

if __name__ == "__main__": 
	# Get the preprocessed data
	dataset_path = '/Users/shyammarjit/Desktop/Brain Computer Interface/Deap Dataset/'
	subject_no = 's01'
	data, label = get_data(dataset_path, subject_no)

	# Multi-Domain feature extraction
	optimal_channels = ['Fp1', 'Fp2', 'F3', 'F4']
	feature = get_features(data, optimal_channels) # 2D matrix, (videos*features)
	feature_names = []
	for i in range(0, feature.shape[1]):
	    feature_names.append('feat_' + str(i+1))

	# create dataframe
	df = pd.DataFrame(data = feature, columns = feature_names)
	# perform min-max scalling on the dataset
	data_arr = df.to_numpy()
	scaler = MinMaxScaler()
	data_arr = scaler.fit_transform(data_arr)
	scalled_data = pd.DataFrame(data_arr, columns=df.columns)

	# perform 10-fold cross validation
	train_data, train_label, test_data, test_label = kfold(scalled_data, label)

	
	#from GAMLP import GA_MLP
	fold = []
	for ifold in trange(0, 10):
	    # convert four-class label to binary class labels
	    trainy_val, trainy_ar = four_to_binary(train_label[ifold])
	    testy_val, testy_ar = four_to_binary(test_label[ifold])
	    
	    #---------------------------------------------------------------------------
	    #                                  Valence
	    #---------------------------------------------------------------------------
	    # data tuple creation
	    data = dict(trainX=train_data[ifold], testX=test_data[ifold], trainY=trainy_val, testY=testy_val)
	    # perform Genetic Algorithm based feature selection
	    opt_indivisual = GAFS(data)
	    # optimal individual to feature subset
	    features = ind_to_feat(data['trainX'].columns, opt_indivisual)
	    # data tuple creation
	    data = dict(trainX=data['trainX'][features], testX=data['testX'][features],\
	                trainY=trainy_val, testY=testy_val)
	    # perform GA-MLP on the optimal feature subsets
	    y_pred_val = GA_MLP(data)
	    # get the performance
	    acc_val, prec_val, recall_val, f1_val = performance(y_pred_val, data['testY'])
	    
	    #---------------------------------------------------------------------------
	    #                                  Arousal
	    #---------------------------------------------------------------------------
	    # data tuple creation
	    data = dict(trainX=train_data[ifold], testX=test_data[ifold], trainY=trainy_ar, testY=testy_ar)
	    # perform Genetic Algorithm based feature selection
	    opt_indivisual = GAFS(data)
	    # optimal individual to feature subset
	    features = ind_to_feat(data['trainX'].columns, opt_indivisual)
	    # data tuple creation
	    data = dict(trainX=data['trainX'][features], testX=data['testX'][features],\
	                trainY=trainy_ar, testY=testy_ar)
	    # perform GA-MLP on the optimal feature subsets
	    y_pred_ar = GA_MLP(data)
	    # get the performance
	    acc_ar, prec_ar, recall_ar, f1_ar = performance(y_pred_ar, data['testY'])
	    
	    #---------------------------------------------------------------------------
	    #                                  Four Class
	    #---------------------------------------------------------------------------
	    y_pred_four = binary_to_four(y_pred_val, y_pred_ar)
	    acc_four, prec_four, recall_four, f1_four = multi_class_performance(test_label[ifold], y_pred_four)
	    print("Fold-" + str(ifold+1), acc_val, prec_val, recall_val, f1_val, acc_ar, prec_ar,\
	          recall_ar, f1_ar, acc_four, prec_four, recall_four, f1_four)
	    fold.append(np.array([acc_val, prec_val, recall_val, f1_val, acc_ar, prec_ar, recall_ar, f1_ar,\
	     acc_four, prec_four, recall_four, f1_four]))
	print("Mean: ", np.mean(fold, axis = 0))