#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import sklearn.feature_selection as feature_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

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
import sysss
from time import time
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm
# Display progress logs on stdout


from sklearn.model_selection import train_test_split


# In[8]:


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


# # Downloading imputed data

# In[57]:


# x_train = pd.read_csv('train_eng_features.csv', index_col=0)
# x_test = pd.read_csv('test_eng_features.csv', index_col=0)

x_train = pd.read_csv('train_all_eng_features.csv', index_col=0)
x_test = pd.read_csv('test_all_eng_features.csv', index_col=0)

pids = np.unique(x_test['pid'].to_numpy())
x_train = x_train.drop(['pid'], axis = 1)
x_test = x_test.drop(['pid'], axis = 1)

y_train_labels = pd.read_csv(path_train_labels)

y_train_labels_sorted = y_train_labels.sort_values('pid')

y_train_labels = y_train_labels.drop(['pid'], axis = 1)

y_sepsis = y_train_labels_sorted['LABEL_Sepsis'].to_numpy()

#Dataframe for predicted data
columns_predicted = ['pid'] + TESTS + ['LABEL_Sepsis'] + VITALS 
predicted_y_df = pd.DataFrame(columns = columns_predicted)
predicted_y_df['pid'] = pids


# In[58]:


x_train.columns


# In[59]:


categoricals = ['custom_age','custom_hr']
indexes_of_categories = [x_test.columns.get_loc(col) for col in categoricals]


# In[60]:


indexes_of_categories


# Pipeline for the first subtask

# In[61]:


selector = SelectPercentile(percentile=100)
#svc = SVC(C=1, kernel='rbf', class_weight='balanced', probability = True, random_state=42)
logreg = LogisticRegression()
lbm = LGBMClassifier(reg_alpha=15, reg_ambda=15, n_estimators=100)
pipeline = make_pipeline(selector, lbm)


# In[ ]:


# Cross validation
score_summary = []
for i in range(len(y1_label)):
    print("Feature: {}".format(y1_label[i]))
    scores = cross_validate(pipeline, x_train, y1[:,i],
                                scoring='roc_auc',
                                return_train_score=True,
                                cv=5,  # Already include stratified folds
                                return_estimator=True,
                                n_jobs=-1)  # Add parallelism to speed up validation
    print("Test scores:\t", scores['test_score'],
            '\nTrain scores:\t', scores['train_score'])
    print("Test: %0.4f (+/- %0.4f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))
    score_summary.append(scores['test_score'].mean())
    print("Train: %0.4f (+/- %0.4f)" % (scores['train_score'].mean(), scores['train_score'].std() * 2))

    pipeline.fit(x_train, y1[:,i])
    y_pred = pipeline.predict_proba(x_test)[:,1]
    
    prediction[y1_label[i]] = y_pred


# In[14]:


LABELS = TESTS


# # First Task

# In[62]:


score_summary = []
for label in TESTS:
    y_train = y_train_labels_sorted[label].to_numpy()
    print("Feature: {}".format(label))
    scores = cross_validate(pipeline, x_train, y_train,
                                scoring='roc_auc',
                                return_train_score=True,
                                cv=5,  # Already include stratified folds
                                return_estimator=True,
                                n_jobs=-1)  # Add parallelism to speed up validation
    print("Test scores:\t", scores['test_score'],
            '\nTrain scores:\t', scores['train_score'])
    print("Test: %0.4f (+/- %0.4f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))
    score_summary.append(scores['test_score'].mean())
    print("Train: %0.4f (+/- %0.4f)" % (scores['train_score'].mean(), scores['train_score'].std() * 2))

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict_proba(x_test)[:,1]
    print(y_pred)
    
    predicted_y_df[label] = y_pred


# # Second Task

# In[63]:


label = 'LABEL_Sepsis'
selector = SelectPercentile(percentile=100)
#svc = SVC(C=1, kernel='rbf', class_weight='balanced', probability = True, random_state=42)
logreg = LogisticRegression()
lbm = LGBMClassifier(reg_alpha=41, reg_ambda=15, n_estimators=100, categorical_feature=indexes_of_categories, class_weight='balanced')
pipeline = make_pipeline(selector, lbm)

i= -1
print("Feature: {}".format(label))
scores = cross_validate(pipeline, x_train, y_sepsis,
                            scoring='roc_auc',
                            return_train_score=True,
                            cv=5,  # Already include stratified folds
                            return_estimator=True,
                            n_jobs=-1)  # Add parallelism to speed up validation
print("Test scores:\t", scores['test_score'],
        '\nTrain scores:\t', scores['train_score'])
print("Test: %0.4f (+/- %0.4f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))
score_summary[-1] = scores['test_score'].mean()
print("Train: %0.4f (+/- %0.4f)" % (scores['train_score'].mean(), scores['train_score'].std() * 2))

pipeline.fit(x_train, y_sepsis)
y_pred = pipeline.predict_proba(x_test)[:,1]

predicted_y_df[label] = y_pred


# # Third Task

# # Label_RRate

# In[64]:


import sklearn.feature_selection as feature_selection

label = VITALS[0]
y_train = y_train_labels_sorted[label].to_numpy()


selector = SelectPercentile(feature_selection.f_regression, percentile=50)

params = {'n_estimators': 500,
              'max_depth': 10,
              'learning_rate': 0.1,
              'loss': 'ls'}
            
scaler = StandardScaler()
gbr = GradientBoostingRegressor(n_estimators = 100, max_depth=3, max_features=None)
lsvr = LinearSVR()
mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100, ))
lgreg = LGBMRegressor(reg_alpha=200, reg_lambda=100, n_estimators=200)
pipeline = make_pipeline(selector, gbr)

i = 0
print("Feature: {}".format(label))
scores = cross_validate(pipeline, x_train, y_train,
                            scoring='r2',
                            return_train_score=True,
                            cv=5,  # Already include stratified folds
                            return_estimator=True,
                            n_jobs=-1)  # Add parallelism to speed up validation
print("Test scores:\t", scores['test_score'],
          '\nTrain scores:\t', scores['train_score'])
print("Test: %0.4f (+/- %0.4f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))
print("Train: %0.4f (+/- %0.4f)" % (scores['train_score'].mean(), scores['train_score'].std() * 2))

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
predicted_y_df[label] = y_pred


# # LABEL_ABPm

# In[65]:



label = VITALS[1]
y_train = y_train_labels_sorted[label].to_numpy()


selector = SelectPercentile(feature_selection.f_regression, percentile=90)
params = {'n_estimators': 500,
              'max_depth': 10,
              'learning_rate': 0.1,
              'loss': 'ls'}
#iforest = IsolationForest()              
scaler = StandardScaler()
gbr = GradientBoostingRegressor(n_estimators = 200, max_depth=5, max_features=None)
lsvr = LinearSVR()
mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100, ))
lgreg = LGBMRegressor(reg_alpha=200, reg_lambda=100, n_estimators=200)
pipeline = make_pipeline(selector, lgreg)

i = 1
print("Feature: {}".format(label))
scores = cross_validate(pipeline, x_train, y_train,
                            scoring='r2',
                            return_train_score=True,
                            cv=5,  # Already include stratified folds
                            return_estimator=True,
                            n_jobs=-1)  # Add parallelism to speed up validation
print("Test scores:\t", scores['test_score'],
          '\nTrain scores:\t', scores['train_score'])
print("Test: %0.4f (+/- %0.4f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))
print("Train: %0.4f (+/- %0.4f)" % (scores['train_score'].mean(), scores['train_score'].std() * 2))

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
predicted_y_df[label] = y_pred


# In[35]:


x_train.shape


# # LABEL_SpO2

# In[75]:


label = VITALS[2]
y_train = y_train_labels_sorted[label].to_numpy()


selector = SelectPercentile(feature_selection.f_regression, percentile=50)
ridge = Ridge(alpha=500.0)
gbr = GradientBoostingRegressor(n_estimators = 100, max_depth=3, max_features=50)
lsvr = LinearSVR()
mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100, ))
lgreg = LGBMRegressor(reg_alpha=100, reg_lambda=100, n_estimators=600)
pipeline = make_pipeline(gbr)

i = 2
print("Feature: {}".format(label))
scores = cross_validate(pipeline, x_train, y_train,
                            scoring='r2',
                            return_train_score=True,
                            cv=5,  # Already include stratified folds
                            return_estimator=True,
                            n_jobs=-1)  # Add parallelism to speed up validation
print("Test scores:\t", scores['test_score'],
          '\nTrain scores:\t', scores['train_score'])
print("Test: %0.4f (+/- %0.4f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))
print("Train: %0.4f (+/- %0.4f)" % (scores['train_score'].mean(), scores['train_score'].std() * 2))

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
predicted_y_df[label] = y_pred


# # LABEL_Heartrate

# In[76]:


selector = SelectPercentile(feature_selection.f_regression, percentile=80)

label = VITALS[3]
y_train = y_train_labels_sorted[label].to_numpy()

params = {'n_estimators': 500,
              'max_depth': 10,
              'learning_rate': 0.1,
              'loss': 'ls'}
#iforest = IsolationForest()              
scaler = StandardScaler()
ridge = Ridge(alpha=0.1)
gbr = GradientBoostingRegressor(n_estimators = 100, max_depth=3, max_features=None)
lsvr = LinearSVR()
mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100, ))
lgreg = LGBMRegressor(reg_alpha=100, reg_lambda=100, n_estimators=500)
pipeline = make_pipeline(selector, gbr)

i = 3
print("Feature: {}".format(label))
scores = cross_validate(pipeline, x_train, y_train,
                            scoring='r2',
                            return_train_score=True,
                            cv=5,  # Already include stratified folds
                            return_estimator=True,
                            n_jobs=-1)  # Add parallelism to speed up validation
print("Test scores:\t", scores['test_score'],
          '\nTrain scores:\t', scores['train_score'])
print("Test: %0.4f (+/- %0.4f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))
print("Train: %0.4f (+/- %0.4f)" % (scores['train_score'].mean(), scores['train_score'].std() * 2))

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
predicted_y_df[label] = y_pred


# In[77]:


predicted_y_df


# In[78]:


predicted_y_df.to_csv('prediction_06.06.zip', index=False, float_format='%.3f', compression='zip')


# In[ ]:




