#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import pandas as pd
from sklearn import svm, model_selection, metrics, linear_model, tree
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression

from lightgbm import LGBMClassifier, LGBMRegressor
import sklearn.feature_selection as feature_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers import Embedding
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Importing libraries for building the neural network
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras import regularizers      #for l2 regularization
from keras.wrappers.scikit_learn import KerasRegressor 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils import class_weight


# In[10]:


acids = ["R", "H", "K", "D", "E", "S", "T", "N", "Q", "C", "U", "G","P", "A", "I", "L", "M", "F","W", "Y", "V"]
def seq_to_vec(seq):
    """amino acid sequence to a number (index in the array of acids)
       H - 1 index -> 1
    """
    vec = []
    for word in seq:
        vec.append(acids.index(word)) 
    return np.array(vec)

def hot_encoder(elem):
    """amino acid sequence to a hot vector of dim 21 (one in the place of index-number in the array of acids)
       H - 1 index -> 010000000000000000000
    """
    hot_vector = np.zeros(21)
    index = acids.index(elem)
    hot_vector[index] = 1
    return hot_vector

def hot_encoder_group(elem):
    """amino acid sequence to a hot vector of dim 21 (number of group in the place of index-number in the array of acids)
       D - 3 index and belongs to 2-d group; i -> 000200000000000000000;
    """
    hot_vector = np.zeros(21)
    index = acids.index(elem)
    group_index = group_amino(index + 1)
    hot_vector[index] = group_index + 1
    return hot_vector

def seq_to_hot(seq):
    vec = []
    for word in seq:
        vec.append(hot_encoder(word)) 
    return np.array(vec)

def seq_to_hot_group(seq):
    vec = []
    for word in seq:
        vec.append(hot_encoder_group(word)) 
    return np.array(vec)

    
def group_amino(idx):
    """return chemical groupping
       check this: https://commons.wikimedia.org/wiki/File:Amino_Acids-wide.svg
    """
    if (idx == 1 or idx == 2 or idx == 3 or idx == 4 or idx == 5):
        return 0
    elif (idx == 6 or idx == 7 or idx == 8 or idx == 9):
        return 1
    elif (idx == 10 or idx == 11 or idx == 12 or idx == 13):
        return 2
    elif (idx == 14 or idx == 15 or idx == 16 or idx == 17 or idx == 18 or idx == 19 or idx == 20 or idx == 21):
        return 3
    
def acids_to_channels(seq):
    """creates 4*4 feature tensor
       every group corresponds to a special channel. Firstly we encode protein into a seqience of indices
       of amino acids array.
       Eg: 2348
       In every "subdimension - channel" we leave just idxs of elements belonging to this group
       Eg 2 and 3 belongs to the first group, 4 to 2d and 8 to 4-th:
       2300
       0040
       0000
       0008
    """
    amino_seq = seq_to_vec(seq)
    tensor = np.zeros((4, 4))
    for group in range(4):
        for elem in range(4):
            if(group_amino(amino_seq[elem]) == group):
                tensor[group, elem] = amino_seq[elem]
    return tensor
    
def traint_test_split(X,y, size):
    train_length = int(size*X.shape[0])
    return X[:train_length], y[:train_length], X[train_length:], y[train_length:]

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_f1(y_true, y_pred): #taken from old keras source code
    #F1 metric for accuracy
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# In[22]:


X = pd.read_csv("data/train.csv")
X['hot_seq'] = X['Sequence'].transform(lambda x: seq_to_hot(x))
# X['hot_seq'] = X['Sequence'].transform(lambda x: seq_to_hot_group(x))
X_all = X['hot_seq'].to_numpy()
X_all = np.stack(X_all)
X_all = np.reshape(X_all, (X_all.shape[0], -1))
# X_all = X_all.reshape((X_all[0], X_all[1]* X_all[2])
y_all = X['Active'].to_numpy()


# In[8]:



class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_all),
                                                 y_all)


# In[23]:


num_folds = 5
verbosity = 1
no_epochs = 100


# # Cross validation with 5 folds and 3 dense layers (FC network)

# In[24]:


kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
for train, test in kfold.split(X_all, y_all):
    model = Sequential()
    model.add(Dense(500, input_dim = 84, activation='relu'))
#     model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    model.add(Dense(500, activation='relu'))
#     model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
#     model.add(Dense(5000, input_dim = 500, kernel_initializer="normal", activation='relu'))
#     model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',get_f1]) 
#     model.summary()
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    history = model.fit(X_all[train], y_all[train],
              class_weight = class_weights,
              verbose=verbosity,
              epochs=no_epochs,
              validation_data=(X_all[test], y_all[test]))
    score = model.evaluate(X_all[test], y_all[test], verbose=0)
    print(f'F1 score for fold {fold_no}: {score[2]}')
    print('Test loss:', score[0])
#     print('Test accuracy:', score[1])
#     print('Test accuracy F1:', score[2])
    fold_no = fold_no + 1
    


# write answer

# In[ ]:


x_final = pd.read_csv('data/test.csv')
x_final['Sequence'] = x_final['Sequence'].transform(lambda x: seq_to_hot(x))
x_final = np.stack(x_final['Sequence'].to_numpy())
x_final = np.reshape(x_final, (x_final.shape[0], -1))
answer = model.predict_classes(x_final)
np.savetxt("answer.csv", answer, delimiter=",", fmt='%d')


# # Early stopping (doesn't help too much)

# In[ ]:



class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_all),
                                                 y_all)
num_folds = 5
verbosity = 1
no_epochs = 100


# In[ ]:


kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
for train, test in kfold.split(X_all, y_all):
    model = Sequential()
    model.add(Dense(1000, input_dim = 84, activation='relu'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
#     model.add(Dense(5000, input_dim = 500, kernel_initializer="normal", activation='relu'))
#     model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',get_f1]) 
#     model.summary()
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=0, mode='auto')]
    history = model.fit(X_all[train], y_all[train],
              class_weight = class_weights,
              verbose=verbosity,
              epochs=no_epochs,
              validation_data=(X_all[test], y_all[test]), callbacks = callbacks)
    score = model.evaluate(X_all[test], y_all[test], verbose=0)
    print(f'F1 score for fold {fold_no}: {score[2]}')
    print('Test loss:', score[0])
#     print('Test accuracy:', score[1])
#     print('Test accuracy F1:', score[2])
    fold_no = fold_no + 1
    

