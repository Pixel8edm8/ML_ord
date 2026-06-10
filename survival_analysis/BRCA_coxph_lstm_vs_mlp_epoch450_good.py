#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# epoch=450，lstmadd=55, c_indices_lstm= 0.6955930748318787
# epoch=2000，c_indices_mlp= 
"""
Demonstrates how the partial likelihood from a Cox proportional hazards
model can be used in a NN loss function. An example shows how a NN with
one linear-activation layer and the (negative) log partial likelihood as
loss function produces approximately the same predictor weights as a Cox
model fit in a more conventional way.
"""
import datetime
import pandas as pd
import numpy as np
import keras
import torch
#from lifelines import CoxPHFitter
#from lifelines.datasets import load_kidney_transplant

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import theano
from keras.layers import Dropout, Activation, Lambda
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Embedding

from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier
from lifelines.utils import concordance_index
from sklearn.preprocessing import minmax_scale
from keras.initializers import glorot_uniform
from keras.regularizers import l2
####################################################################################################################

kidtx = pd.read_csv('brca_Surv_data_methylation_mRNA_all_lmqcm.csv')
dataX1 =kidtx.drop(["Unnamed: 0","ID","V2.x","V3.x"], axis = 1).values
y = np.transpose(np.array((kidtx["V2.x"], kidtx["V3.x"]))) # V1=time; erged_data33=status



[ m0,n0] = dataX1.shape
dataX = np.asarray(dataX1)
dataX =minmax_scale(dataX ) 
data_methylation=dataX1[:,0:17]
data_mRNA=dataX1[:,17:n0-1]
#dataX=data_mRNA
[ m,n] = dataX.shape
[ m1,n1] = data_methylation.shape
[ m2,n2] = data_mRNA.shape

x=dataX 
dataX = dataX.reshape(m,1,n)

data_methylation = data_methylation.reshape(m1,1,n1)
data_mRNA = data_mRNA.reshape(m2,1,n2)

ytime=np.transpose(np.array(kidtx["V2.x"])) # only V1=time;
ystatus= np.transpose(np.array(kidtx["V3.x"])) #only erged_data33=status
## Build model structure
#model = Sequential()
#model.add(Dense(units = 20, activation = "tanh", use_bias = False, input_shape=[12]))
#model.add(Dense(1, kernel_initializer='normal'))
# Define loss function
# y_true = (n x 2) array with y_true[i, 0] the survival time
#          for individual i and y_true[i, 1] the event indicator
# y_pred = (n x 1) array of linear predictor (x * beta) values
from keras.utils import np_utils
ystatus2= np_utils.to_categorical(ystatus)





###################################################################################################################
def neg_log_pl(y_true, y_pred):
    # Sort by survival time (descending) so that
    # - If there are no tied survival times, the risk set
    #   for event i is individuals 0 through i
    # - If there are ties, and time[i - k] through time[i]
    #   represent all times equal to time[i], then the risk set
    #   for events i - k through i is individuals 0 through i
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    time = K.gather(y_true[:, 0], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
    # For each set of tied survival times, put the sum of the
    # corresponding risk (exp[x * beta]) values at the first
    # position in the sorted array of times while setting other
    # positions to 0 so that the cumsum operation will result                         tied 关联的
    # in each of the positions having the same sum of risks
    for i in range(time.shape[0] - 1, 0, -1):
        # Going from smallest survival times to largest
        if time[i] == time[i - 1]:
            # Push risk to the later time (earlier in array position)
            risk[i - 1] = risk[i - 1] + risk[i]
            risk[i] = 0
    event = K.gather(y_true[:, 1], indices = sorting.indices)
    denom = K.cumsum(risk) #这个函数的功能是返回给定axis上的累计和
    terms = xbeta - K.log(denom)
    loglik = K.cast(event, dtype = terms.dtype) * terms   #cast将x的数据格式转化成dtype
    return -K.sum(loglik)

#np.random.seed(13)



## C_index metric function

def c_index3(month,risk, status):

    c_index = concordance_index(np.reshape(month, -1), -np.reshape(risk, -1), np.reshape(status, -1))

    return c_index#def get_bi_lstm_model():  
#    model=Sequential()
#    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(1,input_dim0), merge_mode='concat'))
#
#    model.add(TimeDistributed(Dense(50, activation='tanh')))
#    model.add(Bidirectional(LSTM(50)))
#    model.add(Dropout(0.2))
#    model.add(Dense(1, activation='linear'))
#  
## Compile model
##    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, \
##        epsilon=None, decay=0.0, amsgrad=False), loss = neg_log_pl)
#    model.compile(optimizer='adam',  loss=neg_log_pl)
#    return model

seed =63
np.random.seed(seed)
kf = KFold(n_splits=10, shuffle=True, random_state=seed)
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
        dataX[train_index], dataX[val_index], y[train_index], y[val_index], ytime[train_index], ytime[val_index], ystatus[train_index],ystatus[val_index]
    
    input_dim =x_train.shape[2]
#    output_dim = x_train.shape[1]
    n1 = y_train.shape[0]
#    k_n = theano.shared(np.asarray(n,dtype=theano.config.floatX),borrow=True)
    k_n = theano.shared(n1,borrow=True)
#    input_dim0 =theano.shared(input_dim,borrow=True)
# Build model structure
    model=Sequential()
#    model.add(Bidirectional(LSTM(50,return_sequences=False,activation='linear'), input_shape=(1,input_dim)))

#    model.add(Bidirectional(LSTM(50,return_sequences=False,activation='linear', kernel_initializer='normal'), input_shape=(1,input_dim)))
    model.add(Bidirectional(LSTM(55,activation='linear', return_sequences=False,\
        kernel_initializer=glorot_uniform(),kernel_regularizer=l2(0.0005),activity_regularizer=l2(0.001)), input_shape=(1,input_dim), merge_mode='concat'))

# #    print(model.layers)
#    model.add(TimeDistributed(Dense(50)))
#    model.add(Bidirectional(LSTM(20)))
#    model.add(Dropout(0.1)) # important
#    model.add(Dropout(0.1))
#    model.add(BatchNormalization())
#    model.add(TimeDistributed(Dense(1, activation='linear')))
    model.add(Dense(1))
# Compile model
#    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, \
#        epsilon=None, decay=0.1), loss = neg_log_pl)
    model.compile(optimizer='adam',  loss=neg_log_pl)
    
#    model = Sequential()
#    model.add(Dense(units =80, kernel_initializer='normal', activation = "tanh", use_bias = False, input_shape=[12]))
#    model.add(Dense(1))
    
#    model=Sequential()
#    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(1,input_dim0), merge_mode='concat'))
#
#    model.add(TimeDistributed(Dense(50, activation='tanh')))
#    model.add(Bidirectional(LSTM(50)))
#    model.add(Dropout(0.2))
#    model.add(Dense(1, activation='liner'))
#
#    
## Compile model
#    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, \
#        epsilon=None, decay=0.0, amsgrad=False), loss = neg_log_pl)
#estimator = KerasRegressor(build_fn= model, epochs=100, batch_size=5)
 
    
#results=cross_val_score(estimator, X, y, cv=10,verbose = 1)  #cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值
#pred_test=estimator.predict()    
# Fit model with the whole dataset as a batch, since the
# partial likelihood depends on all observations
#    estimator = KerasRegressor(build_fn= get_bi_lstm_model, epochs=50, batch_size=n1, verbose=1)
#    kfold1 = KFold(n_splits=10, random_state=seed)
#    results = cross_val_score(estimator, x_train, y_train, cv=kfold1)
#    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    
    hist=model.fit(x_train, y_train, batch_size = n1, epochs = 450)
#    hist=estimator.fit(x_train, y)
#    hist.history['loss']
    
    
    prediction = model.predict(x_test)
    prediction_train_median = model.predict(x_train)
    ypred.extend(prediction)
    ypred_train.extend(prediction_train_median)
#    xtest_original.extend(x_test)
    index2.extend(val_index)
    status_new.extend(ystatus[val_index])
    time_new.extend(ytime[val_index])
#    print(ypred.shape)
    K.clear_session()
    tf.reset_default_graph()
    print(iFold)
    nowTime = datetime.datetime.now()
    print("nowTime: ",nowTime)
    
    c_index2=c_index3( np.asarray(ytime_test),np.asarray(prediction), np.asarray(ystatus_test))
    
    print( c_index2)
    
np.savetxt("brca_prediction919_18lstm2222_epoch400_drop01.csv", ypred, delimiter=",")
np.savetxt("brca_ytime_test919_18lstm2222_epoch400_drop01.csv", time_new, delimiter=",")
np.savetxt("brca_ystatus_test919_18lstm2222_epoch400_drop01.csv", status_new, delimiter=",")
np.savetxt("brca_ypred_train_median919_18lstm2222_epoch400_drop01.csv", ypred_train, delimiter=",")
# Compare to Cox model
#cph = CoxPHFitter()
## CoxPHFitter uses Efron's method for handling tied survival times,
## whereas neg_log_pl uses Breslow's method, so the likelihood
## functions being optimized are not exactly the same
#cph.fit(kidtx, duration_col = "V1", event_col = "erged_data33",show_progress=True, step_size=0.1)
#cph.print_summary(decimals=8)
#cph.plot() #画得是两个变量之间的相关关系值
#plt.show()
#cph.plot_covariate_groups('MEgreenyellow', [0, 5, 10, 15])
#plt.show()
#model.get_weights()
df = pd.read_csv("brca_prediction919_18lstm2222_epoch400_drop01.csv",header=None)    
month=np.asarray(pd.read_csv("brca_ytime_test919_18lstm2222_epoch400_drop01.csv",header=None)) 
status=np.asarray(pd.read_csv("brca_ystatus_test919_18lstm2222_epoch400_drop01.csv",header=None)) 


#df=ypred
#month=time_new
#status=status_new
#df1 = pd.read_csv("WAVE_preTest_scores.csv")    
#dataset_init = np.asarray(df1)    # if only 1 column
#dataX, dataY = create_interval_dataset(df, 1)    #这里的输入数据来源是csv文件

risk=np.asarray(df)
c_indices_lstm = c_index3(month, risk,status)
#np.savetxt("c_indices_nn827.txt", c_indices_mlp, delimiter=",")
np.save("c_indices_nn919_18lstm2222_epoch400_drop01.csv",c_indices_lstm) 
print(c_indices_lstm)
data_lstm=np.load('c_indices_nn919_18lstm2222_epoch400_drop01.csv.npy')
aa=0
##########################################################################################mlp_one
K.clear_session()
tf.reset_default_graph()
seed = 63
np.random.seed(seed)
kf = KFold(n_splits=10, shuffle=True, random_state=seed)
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
    n2 =x_train.shape[1]
#    k_n = theano.shared(np.asarray(n,dtype=theano.config.floatX),borrow=True)
    k_n = theano.shared(n1,borrow=True)
# Build model structure
    
    model = Sequential()
    model.add(Dense(units =50, activation = "linear", use_bias = False, input_shape=[n2]))
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
    model.compile(optimizer=keras.optimizers.Adam(amsgrad=False), loss = neg_log_pl)
#estimator = KerasRegressor(build_fn= model, epochs=100, batch_size=5)
 
    
#results=cross_val_score(estimator, X, y, cv=10,verbose = 1)  #cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值
#pred_test=estimator.predict()    
# Fit model with the whole dataset as a batch, since the
# partial likelihood depends on all observations
    hist=model.fit(x_train, y_train, batch_size = n1, epochs = 1000)
    hist.history['loss']
    prediction = model.predict(x_test)
    prediction_train_median = sum(model.predict(x_train))/n1
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
    nowTime = datetime.datetime.now()
    print("nowTime: ",nowTime)
   # np.savetxt("brca_ypred_train_median919_one_mlp_133.csv", ypred_train, delimiter=",")
    
    c_index2=c_index3( np.asarray(ytime_test),np.asarray(prediction), np.asarray(ystatus_test))
    
    print( c_index2)
    
np.savetxt("brca_prediction919_one_mlp.csv", ypred, delimiter=",")
np.savetxt("brca_ytime_test919_one_mlp.csv", time_new, delimiter=",")
np.savetxt("brca_ystatus_test919_one_mlp.csv", status_new, delimiter=",")
np.savetxt("brca_ypred_train_median919_one_mlp.csv", ypred_train, delimiter=",")


df = pd.read_csv("brca_prediction919_one_mlp.csv",header=None)    
month=np.asarray(pd.read_csv("brca_ytime_test919_one_mlp.csv",header=None)) 
status=np.asarray(pd.read_csv("brca_ystatus_test919_one_mlp.csv",header=None)) 



def c_index3(month,risk, status):

    c_index = concordance_index(np.reshape(month, -1), -np.reshape(risk, -1), np.reshape(status, -1))

    return c_index#def get_bi_lstm_model():  

risk=np.asarray(df)
c_indices_mlp = c_index3(month, risk,status)
#np.savetxt("c_indices_nn827.txt", c_indices_mlp, delimiter=",")
np.save("c_indices_919_one_mlp.csv",c_indices_mlp) 
print(c_indices_mlp)
data_a=np.load('c_indices_919_one_mlp.csv.npy')
aa=0