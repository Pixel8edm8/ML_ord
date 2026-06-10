#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#c-index=0.6090234597939383 leave one out good
"""
Demonstrates how the partial likelihood from a Cox proportional hazards
model can be used in a NN loss function. An example shows how a NN with
one linear-activation layer and the (negative) log partial likelihood as
loss function produces approximately the same predictor weights as a Cox
model fit in a more conventional way.
"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.datasets import load_kidney_transplant
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import theano
from keras.layers import Dropout, Activation, Lambda
from keras import backend as K
##############################
# load dataset
kidtx = pd.read_csv('brca_Surv_data.csv',usecols=['MEgreenyellow',	'MEgreen','MEturquoise'	,'MEmagenta','MEbrown',	'MEred',	'MEpink',	'MEblack',	'MEpurple',	'MEblue',	'MEyellow',	'MEgrey','V1','erged_data33'])
#status=pd.read_csv('brca_Surv_data.csv',usecols=["erged_data33"])

#print (puredata.head())
#dataset= puredata.drop('V1',axis=1)
#dataset2 = puredata['V1']  

########################################

# Use example dataset from lifelines module
#kidtx = load_kidney_transplant()
# First three rows:
#   time  death  age  black_male  white_male  black_female
#0     1      0   46           0           1             0
#1     5      0   51           0           1             0
#2     7      1   55           0           1             0
x = kidtx.drop(["V1", "erged_data33"], axis = 1).values
y = np.transpose(np.array((kidtx["V1"], kidtx["erged_data33"]))) # V1=time; erged_data33=status
sample=y.shape[0]
x = np.asarray(x)
ytime=np.transpose(np.array(kidtx["V1"])) # only V1=time;
ystatus= np.transpose(np.array(kidtx["erged_data33"])) #only erged_data33=status
## Build model structure
#model = Sequential()
#model.add(Dense(units = 20, activation = "tanh", use_bias = False, input_shape=[12]))
#model.add(Dense(1, kernel_initializer='normal'))
# Define loss function
# y_true = (n x 2) array with y_true[i, 0] the survival time
#          for individual i and y_true[i, 1] the event indicator
# y_pred = (n x 1) array of linear predictor (x * beta) values
def neg_log_pl(y_true, y_pred):
    # Sort by survival time (descending) so that
    # - If there are no tied survival times, the risk set
    #   for event i is individuals 0 through i
    # - If there are ties, and time[i - k] through time[i]
    #   represent all times equal to time[i], then the risk set
    #   for events i - k through i is individuals 0 through i
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    time = K.gather(y_true[:, 0], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices)
    risk = K.exp(xbeta)
    # For each set of tied survival times, put the sum of the
    # corresponding risk (exp[x * beta]) values at the first
    # position in the sorted array of times while setting other
    # positions to 0 so that the cumsum operation will result
    # in each of the positions having the same sum of risks
    for i in range(time.shape[0] - 1, 0, -1):
        # Going from smallest survival times to largest
        if time[i] == time[i - 1]:
            # Push risk to the later time (earlier in array position)
            risk[i - 1] = risk[i - 1] + risk[i]
            risk[i] = 0
    event = K.gather(y_true[:, 1], indices = sorting.indices)
    denom = K.cumsum(risk)
    terms = xbeta - K.log(denom)
    loglik = K.cast(event, dtype = terms.dtype) * terms
    return -K.sum(loglik)

seed = 42
np.random.seed(seed)
kf = KFold(n_splits=sample, shuffle=True, random_state=seed)
ypred=[]
ypred_train=[]
xtest_original=[]
status_new=[]
time_new=[]
index2=[]
iFold = 0
for train_index, val_index in kf.split(x):
    iFold = iFold+1
#    train_x, test_x, train_y, test_y,= X[train_index], X[val_index], y[train_index], y[val_index] # 这里的X_train，y_train为第iFold个fold的训练集，X_val，y_val为validation set
    x_train, x_test, y_train, y_test, ytime_train, ytime_test, ystatus_train, ystatus_test =\
        x[train_index], x[val_index], y[train_index], y[val_index], ytime[train_index], ytime[val_index], ystatus[train_index],ystatus[val_index]
    
    n1 = y_train.shape[0]
#    k_n = theano.shared(np.asarray(n,dtype=theano.config.floatX),borrow=True)
    k_n = theano.shared(n1,borrow=True)
# Build model structure
    
    model = Sequential()
    model.add(Dense(units = 50, activation = "tanh", use_bias = False, input_shape=[12]))
    model.add(Dense(1, kernel_initializer='normal'))
    
#    model.add(Dense(10)) 
#    model.add(Activation('relu'))
#    model.add(Dropout(0.2))
#    model.add(Dense(units =1, activation = "tanh"))    
##    model.add(Dense(50, activation="relu"))  
#    model.add(Dense(100, kernel_initializer='normal', input_shape=[12]))    
#model.add(Activation(LeakyReLU(alpha=0.01))) 
#    
#    model.add(Dense(50, activation="relu"))    
#model.add(Activation(LeakyReLU(alpha=0.1))) 
#    model.add(activation('relu'))
#    model.add(Dropout(0.25))
    
#model.add(Activation(LeakyReLU(alpha=0.01))) 
   

 
    
#    model.add(Dense(units = 30, activation = "tanh", use_bias = False, input_shape=[12]))
#    model.add(Dropout(0.25))
#    model.add(Dense(1, kernel_initializer='normal'))
# Compile model
    model.compile(optimizer = "adam", loss = neg_log_pl)
#estimator = KerasRegressor(build_fn= model, epochs=100, batch_size=5)
 
    
#results=cross_val_score(estimator, X, y, cv=10,verbose = 1)  #cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值
#pred_test=estimator.predict()    
# Fit model with the whole dataset as a batch, since the
# partial likelihood depends on all observations
    hist=model.fit(x_train, y_train, batch_size = n1, epochs = 2000)
    hist.history['loss']
    prediction = model.predict(x_test)
    prediction_train_median = np.sum(model.predict(x_train))/n1
    ypred.extend(prediction)
    ypred_train.extend(prediction_train_median)
#    xtest_original.extend(x_test)
    index2.extend(val_index)
    status_new.extend(ystatus[val_index])
    time_new.extend(ytime[val_index])
#    print(ypred.shape)
    print(iFold)
    K.clear_session()
    tf.reset_default_graph()
np.savetxt("brca_prediction828_12.csv", ypred, delimiter=",")
np.savetxt("brca_ytime_test828_12.csv", time_new, delimiter=",")
np.savetxt("brca_ystatus_test828_12.csv", status_new, delimiter=",")
np.savetxt("brca_ypred_train_median828_12.csv", ypred_train, delimiter=",")
# Compare to Cox model
cph = CoxPHFitter()
# CoxPHFitter uses Efron's method for handling tied survival times,
# whereas neg_log_pl uses Breslow's method, so the likelihood
# functions being optimized are not exactly the same
cph.fit(kidtx, duration_col = "V1", event_col = "erged_data33",show_progress=True, step_size=0.1)
cph.print_summary(decimals=8)
cph.plot() #画得是两个变量之间的相关关系值
plt.show()
cph.plot_covariate_groups('MEgreenyellow', [0, 5, 10, 15])
plt.show()
model.get_weights()
aa=0