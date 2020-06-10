#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
# from mice import Mice
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
#new packages
# from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
# from sklearn.impute import KNNImputer
import matplotlib
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer, ColumnTransformer
from matplotlib.pyplot import figure
from numpy import savetxt
from numpy import loadtxt
import sklearn.metrics as metrics
from sklearn import svm
import mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector
import my_utils
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm


from sklearn.model_selection import train_test_split


# In[19]:


####################LAbels of Tests###############################
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

path_train_features = "data/train_features.csv"
path_train_labels = "data/train_labels.csv"
path_test_features = "data/test_features.csv"
x_test_raw = pd.read_csv(path_test_features)


columns_labels_first = ["BaseExcess", "Fibrinogen", "AST", "Alkalinephos", "Bilirubin_total", "Lactate", "TroponinI", "SaO2", "Bilirubin_direct", "EtCO2"]
columns_labels_second = ["Sepsis"]
columns_labels_third = ["RRate", "ABPm", "SpO2", "Heartrate"]

columns_params_test = [col for col in x_train_raw.columns if col not in  columns_labels_first + columns_labels_third + ['Time', 'pid', 'Age']]

interest_col = columns_labels_first + columns_labels_second + columns_labels_third + ['Time', 'pid', 'Age']

x_train_raw = pd.read_csv(path_train_features)

y_train_labels = pd.read_csv(path_train_labels)


# In[21]:


s = np.array([3,4,5])
np.min(s)


# # Feature Engineering

# I have created features of 
# 
# 1) Number of not none (number of tests) for the test labels and parameters (Bilirubin etc)
# 
# 2) The time of the last test for the 1-st subtask
# 
# 3) Mean of values for all values
# 
# 4) Min and Max for values in 3-d subtask
# 
# 5) Categorical variables: age and heart
# 
# 

# In[28]:


def raw_to_imputed(pid, data_raw, label_raw):
    #creates imputed values to a new pad.dataframe for all pid
    data_patient = data_raw.loc[data_raw['pid'] == pid, label_raw].to_numpy()
    data_patient = data_patient.reshape(-1, 1)
    imp = IterativeImputer(missing_values=np.nan)
    imputed_data_patient = imp.fit_transform(data_patient)
    if(imputed_data_patient.shape[1] == 0):
        data_raw.loc[data_raw['pid'] == pid, label_raw] = np.zeros(12)
    else:
        data_raw.loc[data_raw['pid'] == pid, label_raw] = imputed_data_patient
            
def imputation_pid(data_raw, pid, label_raw):
    '''imputes data for patient with pid during all time using sklearn Iterative Imputer(12 days)
    '''
    data_patient = data_raw.loc[data_raw['pid'] == pid, label_raw].to_numpy()
    data_patient = data_patient.reshape(-1, 1)
    imp = IterativeImputer(missing_values=np.nan)
    imputed_data_patient = imp.fit_transform(data_patient)
    if(imputed_data_patient.shape[1] == 0):
        return np.zeros(12)
    else:
        return imputed_data_patient
            
def _get_value(data, pid: int, Label: str):
    """get value in the dataframe data at pid and Label
    """
    return data.loc[data['pid'] == pid, Label].to_numpy()

def mean_generator(pid, old_data, new_data, old_name, new_name):
    '''calculates mean of features (old_name) from impured raw data (old_data)
       and puts them to a new engeneered feature set (new_name)
    '''
    new_values = np.mean(imputation_pid(old_data, pid, old_name))
    new_data.loc[new_data['pid'] == pid, new_name] = new_values

def max_generator(pid, old_data, new_data, old_name, new_name):
    '''calculates max of features (old_name) from impured raw data (old_data)
       and puts them to a new engeneered feature set (new_name)
    '''
    new_values = np.max(imputation_pid(old_data, pid, old_name))
    new_data.loc[new_data['pid'] == pid, new_name] = new_values
    
def min_generator(pid, old_data, new_data, old_name, new_name):
    '''calculates min of features (old_name) from impured raw data (old_data)
       and puts them to a new engeneered feature set (new_name)
    '''
    new_values = np.min(imputation_pid(old_data, pid, old_name))
    new_data.loc[new_data['pid'] == pid, new_name] = new_values
    

def num_not_none_generator(pid, old_data, new_data, old_name, new_name):
    '''calculates number of NONE in feature (old_name) from impured raw data (old_data)
       and puts them to a new engeneered feature set (new_name) - (new_data)
    '''
    #for columns from 1-st subtask
    new_values = 12 - np.sum(np.isnan(old_data.loc[old_data['pid'] == pid, old_name].to_numpy())) 
    new_data.loc[new_data['pid'] == pid, old_name] = new_values
    
def num_last_not_none_generator(pid, old_data, new_data, old_name, new_name):
    '''calculates the time(day) of last not NONE in feature (old_name) from impured raw data (old_data)
       and puts them to a new engeneered feature set (new_name) - (new_data)
    '''
    #for columns from 1-st subtask 
    #put the day of last test
   
    none_id = np.isnan(old_data.loc[old_data['pid'] == pid, old_name].to_numpy())
    result = np.where(none_id == False)
    if (result[0].shape[0] == 0):
        new_data.loc[new_data['pid'] == pid, new_name] = 0
    else:
        new_data.loc[new_data['pid'] == pid, new_name] = old_data.loc[old_data['pid'] == pid, 'Time'].to_numpy()[result[0][-1]]

            
def mean_gen_simple(data, new_data, name, new_name):
    '''attempt to use groupby feature
    '''
    new_data[new_name] = data.groupby(['pid'])[name].transform(lambda x: np.mean(x))

def max_gen_simple(data, new_data, name, new_name):
    '''max of data
    '''
    new_data[new_name] = data.groupby(['pid'])[name].transform(lambda x: np.max(x))
    
def min_gen_simple(data, new_data, name, new_name):
    '''min of data
    '''
    new_data[new_name] = data.groupby(['pid'])[name].transform(lambda x: np.min(x))
    
def feature_engineer_age(train):
    '''generates "categorical variables of Age"
    '''
    train.loc[train['Age'] >=65, 'custom_age'] = 2
    train.loc[train['Age'] <20, 'custom_age'] = 0
    train.loc[(train['Age'] >=20) & (train['Age'] <65), 
            'custom_age'] = 1
    return train


def feature_engineer_hr(train, label_old):
    #label_old - Mean_Heartrate
    '''generates "categorical variables of Heart Rate"
    '''
    train.loc[train[label_old] >= 100,
            'custom_hr'] = 1
    train.loc[train[label_old] < 100,
            'custom_hr'] = 0
    return train


#for columns from first task num_not_none_generator and num_last_not_none_generator
#for columns from the third task mean_generator


#generate pd dataframe with pid and features (mean for third task, number not none,last not tone for first task)
#use this x for sepsis in 2 task
#     values_patient = data.loc[data['pid'] == pid, Label].to_numpy()

def dataset_preparation(data, name_eng_features):
    
    
    #creation of an empty matrix
    pids = np.unique(data['pid'].to_numpy())
    data_eng = pd.DataFrame(columns = ['pid'])
    data_eng['pid'] = pids
    #add age column
    age_data = [_get_value(data, pid, "Age")[0] for pid in pids]
    data_eng['Age'] = age_data
    
    #defining columns for future feature calculation
    not_none_columns = columns_labels_first + columns_params_test
    mean_features = columns_labels_first + columns_labels_third + columns_params_test
    max_min_features = columns_labels_third
    
    for pid in pids:

    #features of number not none & time of the last not none
    
#     not_none_columns = columns_labels_first 
        for label in not_none_columns:
            new_label_1 = "Not_none_" +  label
            new_label_2 = "Last_n_none_" + label
            num_last_not_none_generator(pid, data, data_eng, label, new_label_1)
            num_not_none_generator(pid, data, data_eng, label, new_label_2)


        #calculation of mean
        for label in mean_features:
            new_label = "Mean_" + label
            mean_generator(pid, data, data_eng, label, new_label)


        for label in max_min_features:
            new_label = "Min_" + label
            max_generator(pid, data, data_eng, label, new_label)
            new_label = "Max_" + label
            min_generator(pid, data, data_eng, label, new_label)

        
        feature_engineer_age(data_eng)
        feature_engineer_hr(data_eng, 'Mean_Heartrate')

    
    data_eng.to_csv(name_eng_features)


# write features to csv files

# In[ ]:


dataset_preparation(x_train_raw,  'train_all_eng_features.csv')
dataset_preparation(x_test_raw,  'test_all_eng_features.csv')


# check what we have generated

# In[32]:


x_train_check = pd.read_csv('train_all_eng_features.csv', index_col=0)


# In[34]:


x_train_check.shape

