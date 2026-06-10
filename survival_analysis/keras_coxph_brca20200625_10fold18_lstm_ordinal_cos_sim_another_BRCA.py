#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# x_train+2*methylation , epoch=550，c_index_lstm=0.6535269709543569
# x_train+10*methylation , epoch=550，c_index_lstm=
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
#from lasagne.nonlinearities import rectify,selu
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import theano
from keras.layers import Dropout, Activation, Lambda,Input,Layer
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Embedding

from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier
from lifelines.utils import concordance_index
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import minmax_scale
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from multiprocessing.dummy import Pool as ThreadPool

from keras.layers import  Concatenate
from keras.models import Model
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
#from keras_contrib.layers import CRF
#import FMLayer
##############################
##############################
import os
print (os.path.dirname(__file__))
source_path =os.path.dirname(__file__) + "/data/"
file1 =source_path + "BRCA.csv"
kidtx = pd.read_csv(file1,header=None)
kidtx0=kidtx.ix[:,~((kidtx==0).all())] #去掉全0列
#print (puredata.head())
#dataset= puredata.drop('V1',axis=1)
#dataset2 = puredata['V1']  
kidtx0= np.asarray(kidtx0)
########################################
m0=kidtx0.shape[1]
# Use example dataset from lifelines module
#kidtx = load_kidney_transplant()
# First three rows:
#   time  death  age  black_male  white_male  black_female
#0     1      0   46           0           1             0
#1     5      0   51           0           1             0
#2     7      1   55           0           1             0
#x = kidtx.drop(["0", "1","2"], axis = 1).values
x = kidtx0[:,3:m0]



y =kidtx0[:,0:2]# np.transpose(np.array((kidtx["0"], kidtx["1"]))) # V1=time; erged_data33=status

x = np.asarray(x)
#x =minmax_scale(x )
[ m,n] = x.shape
 
dataX = x.reshape(m,1,n)

ytime=y[:,0] # only V1=time;
ystatus=y[:,1]# np.transpose(np.array(kidtx["1"])) #only erged_data33=status


from keras.utils import np_utils
ystatus2= np_utils.to_categorical(ystatus)

#ytime=minmax_scale(ytime)  
## Build model structure
#model = Sequential()
#model.add(Dense(units = 20, activation = "tanh", use_bias = False, input_shape=[12]))
#model.add(Dense(1, kernel_initializer='normal'))
# Define loss function
# y_true = (n x 2) array with y_true[i, 0] the survival time
#          for individual i and y_true[i, 1] the event indicator
# y_pred = (n x 1) array of linear predictor (x * beta) values
def neg_log_pl(y_true, y_pred):
    xbeta=y_pred
    event =y_true
    risk = K.exp(xbeta)
# #####################################################################  
##  ###########################################################################  
####    
#    for i in range(y_true.shape[0] - 1, 0, -1):
#        # Going from smallest survival times to largest
#        if  risk[i] >  risk[i - 1]:
#            # Push risk to the later time (earlier in array position)
#            risk[i - 1] = risk[i - 1] + risk[i]
#            risk[i] = 0
####            
###            
##            
#            
#######################################################################################################
# ######################################################################################           
    denom = K.cumsum(risk) #这个函数的功能是返回给定axis上的累计和
    terms = xbeta - K.log(denom)#+1e-8)
    loglik = K.cast(event[:,-1], dtype = terms[:,-1].dtype) * terms[:,-1] # K.cast(event, dtype = terms.dtype) * terms   #cast将x的数据格式转化成dtype   #cast将x的数据格式转化成dtype
#    loglik =K.cast(event, dtype = terms.dtype) * terms
#    loglik =K.dot(K.cast(K.transpose(1-event), dtype = terms.dtype),terms)
    loss1= -K.sum(loglik)
#    loss1= -loglik
    return tf.reduce_mean(loss1)
#    # Sort by survival time (descending) so that
#    # - If there are no tied survival times, the risk set
#    #   for event i is individuals 0 through i
#    # - If there are ties, and time[i - k] through time[i]
#    #   represent all times equal to time[i], then the risk set
#    #   for events i - k through i is individuals 0 through i
#    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
#    time = K.gather(y_true[:, 0], indices = sorting.indices)
#    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
#    risk = K.exp(xbeta)
#    # For each set of tied survival times, put the sum of the
#    # corresponding risk (exp[x * beta]) values at the first
#    # position in the sorted array of times while setting other
#    # positions to 0 so that the cumsum operation will result                         tied 关联的
#    # in each of the positions having the same sum of risks
# #####################################################################  
#  ###########################################################################  
def LOSS_L2(y_true, y_pred):
#    MAX_SEQ_LEN=1
#    BATCH_SIZE=int(k_n.get_value())
#    L2_NORM = 0.001
    
   
#    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
##    time = K.gather(y_true[:, 0], indices = sorting.indices)
#    xbeta = K.gather(y_pred, indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
#    risk = K.exp(xbeta)
#    event =K.gather(y_true[:,1], indices = sorting.indices)
    xbeta=y_pred
    event =y_true[:,1]
    risk = K.exp(xbeta)
#    self.preds = preds
    final_dead_rate = tf.nn.softplus(xbeta)
    final_survival_rate=1.0-final_dead_rate
    predict=K.stack([final_survival_rate, final_dead_rate])
    cross_entropy = -K.sum(K.cast(event, dtype = (final_dead_rate).dtype)*(K.log(final_dead_rate)[:,-1]))
#    - K.cumsum( event*K.log(final_dead_rate))
    cost=cross_entropy
    
#    final_survival_rate=tf.subtract(tf.constant(1.0, dtype=tf.float32), final_dead_rate)
#    predict = tf.transpose(tf.stack([final_survival_rate, final_dead_rate]), name="predict")
##    predict =predict[-1,:,:]
#    cross_entropy = -tf.reduce_sum( event*tf.log(tf.clip_by_value(predict,1e-10,1.0)))
#    tvars = tf.trainable_variables()  #tf.trainable_variables 返回所有 当前计算图中 在获取变量时未标记 trainable=False 的变量集合
#    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars ]) * L2_NORM
#    cost = tf.add(cross_entropy, lossL2, name = "cost")  / BATCH_SIZE

    
#    Loss2=K.categorical_crossentropy( event, xbeta)
    Loss=cost
    return Loss
#######################################################################################################
# ######################################################################################           
#    event = K.gather(y_true[:, 1], indices = sorting.indices)
#    denom = K.cumsum(risk) #这个函数的功能是返回给定axis上的累计和
#    terms = xbeta - K.log(denom)
#    loglik = K.cast(event, dtype = terms.dtype) * terms   #cast将x的数据格式转化成dtype
#    return -K.sum(loglik)

#numba  Numba 可以实现提升速度但又不需要改写部分代码为其他编程语言。
    
#from numba import vectorize, int64
# 
#@vectorize([int64(int64)], target='parallel')
    
#from numba import jit
#@jit(nopython=True)
#@jit
###############################################################################################
###############################################################################################
def unique_set(Y_hazard):

    a1 = Y_hazard#.numpy()
#    print('Y_hazard:',Y_hazard)
    # Get unique times
    t, idx = np.unique(a1, return_inverse=True)

    # Get indexes of sorted array
    sort_idx = np.argsort(a1)
#    print(sort_idx)
    # Sort the array using the index
    a_sorted =a1[sort_idx]# a1[np.int(sort_idx)]# a[tf.to_int32(sort_idx)]#
#    print('a_sorted:', a_sorted)
    # Find duplicates and make them 0
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))

    # Difference a[n+1] - a[n] of non zero indexes (Gives index ranges of patients with same timesteps)
    unq_count = np.diff(np.nonzero(unq_first)[0])

    # Split all index from single array to multiple arrays where each contains all indexes having same timestep
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))

    return t, unq_idx

###########################################################################################
###########################################################################################
def ordinal_loss_grad_test (Y_true, score, epsilon=1e-8):
    Y_true=Y_true.numpy()
    Y_hazard0=Y_true[:,0]
    Y_survival=Y_true[:,1]
#            Y_survival_train1=tf.reshape(Y_survival_train, [-1, batch_size,1])
#            Y_survival_train2=Y_survival_train1[-1,:,:] 
    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_true.shape[0],1])
    Y_hazard=Y_hazard1M[-1,:,-1]
    
    t, H = unique_set(Y_hazard) # t:unique time. H original index.
    score=score.numpy()
#    Y_survival=Y_survival.numpy()
#    risk=np.exp(score)
    Y_hazard=Y_hazard.numpy()
    actual_event_index = np.nonzero(Y_survival)[0]
    H = [list(set(h) & set(actual_event_index)) for h in H]
    n = [len(h) for h in H]
    
    matrix_ones = np.zeros([Y_hazard.shape[0], Y_hazard.shape[0]])
    total = 0.0
    for j in range(len(t)):
#        total_1 = np.sum(np.log(np.absolute(score[H[j]] + epsilon)))
        total_1 = score[H[j]]
        m = n[j]
        total_2 = 0.0
        for i in range(m):
            matrix_ones[H[j],sum(H[j:],[])]=1
            risk_more_j=np.exp(score[sum(H[j:],[])])
            risk_j=np.exp(score[H[j]])
            
            rec=risk_j-risk_more_j
            rec2=np.maximum(0,1-rec)
            matrix_ones[H[j],sum(H[j:],[])]=rec2[:,-1]
            subtotal = np.sum(matrix_ones[H[j],sum(H[j:],[])])
            
#            subtotal = np.sum(np.exp(score[sum(H[j:],[])]) )
#            subtotal = np.log(np.absolute(subtotal + epsilon))
            total_2 = total_2 + subtotal
        total = total + total_2
    dloss=np.sum(matrix_ones,axis=0)
    return tf.to_float(dloss)  
  

def ordinal_loss2 ( Y_hazard, Y_survival ,score, epsilon=1e-8):
    t, H = unique_set(Y_hazard) # t:unique time. H original index.
    score=score.numpy()
    Y_survival=Y_survival.numpy()
    Y_hazard,=Y_hazard.numpy()
    actual_event_index = np.nonzero(Y_survival)[0]
    
    acc_pair = []
#    for i in idx_d:
    for i in range(Y_hazard.shape[0]):
        all_j =  range(Y_hazard.shape[0])[Y_hazard > Y_hazard[i]]
        acc_pair += [(i,j) for j in all_j]
    
#    H = [list(set(h) & set(actual_event_index)) for h in H]
#    n = [len(h) for h in H]
#
#    total = 0.0
#    for j in range(len(t)):
##        total_1 = np.sum(np.log(np.absolute(score[H[j]] + epsilon)))
#        total_1 = score[H[j]]
#        m = n[j]
#        total_2 = 0.0
#        for i in range(m):
#            subtotal = np.sum(np.exp(score[sum(H[j:],[])]) )
#            subtotal = np.log(np.absolute(subtotal + epsilon))
#            total_2 = total_2 + subtotal
#        total = total + total_1 - total_2
#    return np.float32(total)    

#######################################################################################
    
def neg_log_pl_1(y_true, y_pred):
    
    xbeta=y_pred
    ytime=y_true[:,0]
    event =y_true[:,1]
    risk = K.exp(xbeta)
    for i in range(ytime.shape[0]-1, 0, -1):
        rec_ij =ytime[i] -ytime[i-1]
        
        
#       if ytime[i]< ytime[i-1]:
            
        
        
        
    
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    ytime = K.gather(y_true[:, 0], indices = sorting.indices)
    yevent = K.gather(y_true[:, 1], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
    matrix_risk = tf.zeros([int(k_n.get_value())],tf.float32)
    matrix_I = tf.zeros([int(k_n.get_value())],tf.float32)
    matrix_max= tf.zeros([int(k_n.get_value())],tf.float32)
    
    kk_ytime_train =k_ytime_train.get_value()
    
#    weights = K.cast(K.abs(K.argmax(ytime, axis=0) - K.argmax(xbeta, axis=0))/(int(k_n.get_value()) - 1), dtype='float32')
    
    for i in range(ytime.shape[0] - 1, 0, -1):
        # Going from smallest survival times to largest
#        tf.cond(tf.greater_equal(ytime[i-1], ytime[i]),  risk[i - 1]: (risk[i - 1] + risk[i]),  risk[i - 1]: 0)
#        risk[i - 1]=tf.where(tf.equal(ytime[i-1], ytime[i]), risk[i - 1] + risk[i],  risk[i - 1])
#        risk_i,risk_i-1 = risk_ab( risk[i], risk[i - 1])
#        sess = tf.Session()
#        array_i = np.array(kk_ytime_train[i])
#        array_i_1 = np.array(kk_ytime_train[i-1])
#        feed_dict = {a: array_i, b: array_i_1}
#        tile_b_value = sess.run(tile_tensor_b, feed_dict = feed_dict)
#        print(tile_b_value)
#        def true_proc():
#            risk[i - 1] = risk[i - 1] + risk[i]
#            return risk[i - 1]
#		
#        def false_proc():
#	         risk[i] = 0
#	         return risk[i]
#        def true_proc():
#            result=2
#            return result
#		
#        def false_proc():
#            result=3
#            return result
        a0=tf.constant(0,dtype=tf.float32)      
        a1=K.cast(risk[i - 1],dtype=tf.float32)
        a2=K.cast(risk[i],dtype=tf.float32)
        a3=K.cast(a1+a2,dtype=tf.float32)
        risk_i_1= tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: a3, lambda:a1)
        risk_i= tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: a0, lambda:a2)
        
#        part1 = matrix_risk[:i-1]
#              
#        part2 = matrix_risk[i+1:]
#        val_i_1= risk_i_1
#        val_i = risk_i
#        matrix_risk2=tf.concat([part1,val_i_1,val_i,part2], axis=0)
#        matrix_risk=K.cast(risk,dtype=tf.float32)
        # 生成一个one_hot张量，长度与tensor_1相同，修改位置为1
        shape = risk.get_shape().as_list()
        one_hot_i = tf.one_hot(i,shape[0],dtype=tf.float32)
        one_hot_i_1 = tf.one_hot(i-1,shape[0],dtype=tf.float32)
       # 做一个减法运算，将one_hot为一的变为原张量该位置的值进行相减
        new_tensor = matrix_risk+risk_i_1 * one_hot_i_1
        matrix_risk = new_tensor+ risk_i * one_hot_i
        
        
        
        rec_ij = a2 - a1
        b1=tf.constant(1,dtype=tf.float32) 
        max_rec_ij= tf.cond(tf.less( rec_ij,  b1), lambda:1-rec_ij, lambda:a0)
        
        Iij=tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: b1, lambda:a0)
        mul_ij=Iij * max_rec_ij
        
        one_hot_I = tf.one_hot(i,shape[0],dtype=tf.float32)
        
        matrix_I = matrix_I+mul_ij*one_hot_I
        matrix_max=matrix_max+max_rec_ij*one_hot_I

            
#    cost2 = ( K.sum(K.dot(matrix_I,matrix_max)) )  
#    cost2 =( 1+weights) * ( K.sum(matrix_I) )     
    cost2 =( K.sum(matrix_I) )    
    cost3 = ( K.sum(matrix_risk) )  
    
#    cost2 =(1.0 + weights) * ( K.sum(matrix_I) )    
#    cost3 =(1.0 + weights) * ( K.sum(matrix_risk) )    
    
#        new_tensor = tf.concat([part1,val,part2], axis=0)

#        a1_list = []
#        a1_list.append(risk_i_1)
#        matrix_risk[i - 1] = tf.stack(a1_list)
#        risk[i - 1]=K.cast( risk_i_1,dtype=risk_i_1.dtype)
#        
#        risk[i]=K.cast( risk_i_1,dtype=risk_i.dtype)
#        risk_i_1= tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: risk[i - 1], lambda:(risk[i - 1] + risk[i]))
#        with tf.Session() as sess: 
#            sess.run(tf.global_variables_initializer()) 
#            y1=sess.run([risk_i_1]) 
#            print(y1)
#        risk_i_1= tf.cond(tf.greater_equal( kk_ytime_train[i],  kk_ytime_train[i-1]), true_fn = true_proc, false_fn = false_proc)
#        n1 = ytime[i]!=tf.constant(0)
#        n2 = ytime[i-1]!=tf.constant(0)
#        if kk_ytime_train[i]<=kk_ytime_train[i-1]:
#            # Push risk to the later time (earlier in array position)
#            risk[i - 1] = risk[i - 1] + risk[i]
#            risk[i] = 0
    
 
#    N=ytime.shape[0]
#    R_matrix = tf.zeros([N,N],tf.int32)
#    
#   
#    
#    for i in range(N - 1, 0, -1):
#        # Going from smallest survival times to largest
#        if ytime[i] == ytime[i - 1]:
#            # Push risk to the later time (earlier in array position)
#            risk[i - 1] = risk[i - 1] + risk[i]
#            risk[i] = 0
    n_observed = tf.reduce_sum(yevent,0)
    ytime_indicator = R_set(ytime)
    
    
#    R_matrix1=add_rankLoss_numba(ytime)
    
#    pool = ThreadPool()
#    R_matrix1=pool.map(add_rankLoss_numba(ytime))
#    pool.close()
#    pool.join()
    
#    for i in range(N):
#        for j in range(N):
#            R_matrix[i,j] = y_true[j] >= y_true[i]
            
   
	###if gpu is being used
#    if torch.cuda.is_available():
#        ytime_indicator = ytime_indicator.cuda()
	###
#    risk_set_sum = (ytime_indicator)*(K.exp(y_pred)) 
#    risk_set_sum =  K.sum(K.cast(ytime_indicator, dtype = y_pred.dtype)*(K.exp(xbeta)),axis=1)
    risk_set_sum =  K.sum(risk)
#    risk_set_sum =  K.sum(K.cast(K.exp(xbeta), dtype = ytime_indicator.dtype)*(ytime_indicator),axis=1,dtype=float32)
    
    diff = xbeta - K.log(risk_set_sum)
#    sum_diff_in_observed = K.cast(K.transpose(diff), dtype = yevent.dtype)*(yevent)
    sum_diff_in_observed = K.cast(yevent, dtype = diff.dtype)*(diff)
#    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    cost = (- K.sum(sum_diff_in_observed))
#    cost = (- K.sum(sum_diff_in_observed / n_observed))
    
    
#    loss = (cost + regularize_layer_params(self.network,l1) * L1_reg
#            + regularize_layer_params(self.network, l2) * L2_reg
#        )
    
    λ1=1
    λ2=-0.05
    λ3=-0.05
    
    return(λ1*cost+λ2*cost2+λ3*cost3)

 
# Custom loss layer
#自定义损失层，损失包含两个部分：图片的重构误差（均方差Square Loss）以及隐变量与单位高斯分割之间的差异（KL-散度KL-Divergence Loss）。
###########################################################################################
def Ordinal_Loss0 (y_true, y_pred):
     
   
#    Y_hazard=k_ytime_train.get_value()
#    Y_survival=k_ystatus_train.get_value()
#   
#    t, H = unique_set(Y_hazard) # t:unique time. H original index.
#    
##    Y_survival=Y_survival.numpy()
##    risk=np.exp(score)
##    Y_hazard=Y_hazard.numpy()
#    actual_event_index = np.nonzero(Y_survival)[0]
#    H = [list(set(h) & set(actual_event_index)) for h in H]
#    n = [len(h) for h in H]
    
    
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    ytime = K.gather(y_true[:, 0], indices = sorting.indices)
    yevent = K.gather(y_true[:, 1], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
    matrix_risk = tf.zeros([int(k_n.get_value())],tf.float32)
#    matrix_I = tf.zeros([int(k_n.get_value())],tf.float32)
    matrix_max= tf.zeros([int(k_n.get_value())],tf.float32)
    
#    kk_t =k_ordinal_t.get_value()
#    kk_n =k_ordinal_n.get_value()
#    Hj =k_ordinal_H.get_value()
    a1=tf.constant(1,dtype=tf.float32) 
    for j in range(len(Hj)):
#        print('j:',j)
#        m=kk_n[j]
#        Hj=sum(H[j:],[])
        matrix_j = tf.zeros([int(k_n.get_value())],tf.float32)
        for i in range(1):
         # 生成一个one_hot张量，长度与tensor_1相同，修改位置为1
            for ii in  range(j,len(Hj)):
#                print('ii:',ii)
                risk_more_j=xbeta[Hj[ii]]
                risk_j=xbeta[Hj[j]]
            
                rec= a1-K.exp(risk_j-risk_more_j)
#                rec2=tf.maximum(0.,rec)
            
                shape = risk.get_shape().as_list()
#                one_hot_j = tf.one_hot(H[j],shape[0],dtype=tf.float32)
                one_hot_more_j = tf.one_hot(Hj[ii],shape[0],dtype=tf.float32)
#                one_hot_more_j =tf.reduce_sum(one_hot_more_j0,axis=0)
               # 做一个减法运算，将one_hot为一的变为原张量该位置的值进行相减
#                new_tensor = matrix_risk+risk_j * one_hot_j
                matrix_j = matrix_j+ rec * one_hot_more_j
        #        tf.reduce_sum(tf.one_hot(sum(H[13:],[]),n1,dtype=tf.float32),axis=0)
    matrix_risk= matrix_risk+ matrix_j 
    cost2 = K.sum(matrix_risk)#/(len(Hj))
    return cost2 
    
    
######################################################################################################

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
#############################################################3
#Training and quality measurement helper functions
def loss_fn(y_true, y_pred):
    """Computes loss function using softmax activation function and categorical cross entropy.
    """

    return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=y_pred, labels=y_true))

def get_avg_acc_loss(dataset, model):
    """Helper function for measuring average loss and accurancy of model.
    """

    def get_accuracy(y_true, y_pred):
        y_pred = tf.argmax(y_pred, 1)
        equality = tf.equal(y_pred, tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        return accuracy
    
    avg_acc = tf.Variable(0,dtype=tf.float32)
    avg_loss = tf.Variable(0,dtype=tf.float32)
    for (batch, (images, labels)) in enumerate(dataset):
        avg_acc = tf.math.add(avg_acc,get_accuracy(labels, model(images)))
        avg_loss = tf.math.add(avg_loss,loss_fn(labels, model(images)))
    
    return avg_acc/(batch+1), avg_loss/(batch+1)

#def get_grad_cos_sim(grad1, grad2):
#    """Computes cos simillarity of gradients after flattening of tensors.
#    
#    It hasn't been stated in paper if batch normalization is considered as model trainable parameter,
#    but from my perspective only convolutional layer's cosine similarities should be measured.
#    """
#    
#    grad1 = tf.concat(
#        [tf.reshape(x,shape=(-1,)) 
#         for i, x in enumerate(grad1) 
#         if i not in batch_norm_ind+task_specific_ind], 
#        axis = 0)
#    grad2 = tf.concat(
#        [tf.reshape(x,shape=(-1,)) 
#         for i, x in enumerate(grad2) 
#         if i not in batch_norm_ind+task_specific_ind], 
#        axis = 0)
#    
#    # perform min(max(-1, dist),1) operation for eventual rounding errors (there's about 1 every epoch)
#    dist = 1 - tf.losses.cosine_distance(grad1,grad2,axis = 0)
#    dist = tf.minimum(tf.maximum(-1, dist),1)
#    
#    return dist

def get_grad_cos_sim(grad1, grad2, sep_ind):
    """Computes cos simillarity of gradients after flattening of tensors.
    """
#    grad12= tf.concat([tf.reshape(grad1[1:2],shape=(-1,)),tf.reshape(grad1[2:-1],shape=(-1,))],axis=0)
#    grad22= tf.concat([tf.reshape(grad2[1:2],shape=(-1,)),tf.reshape(grad2[2:-1],shape=(-1,))],axis=0)
    
#    grad1 = tf.concat([tf.reshape(x,shape=(-1,)) for x in enumerate(grad1[1:-2])], axis = 0)
#    grad2 = tf.concat([tf.reshape(x,shape=(-1,)) for x in enumerate(grad2)], axis = 0)
    grad1 = tf.concat([tf.reshape(x,shape=(-1,)) for x in grad1[:sep_ind]], axis = 0)
    grad2 = tf.concat([tf.reshape(x,shape=(-1,)) for x in grad2[:sep_ind]], axis = 0)
    dist = 1 - tf.losses.cosine_distance(grad1,grad2,axis = 0)
    dist = tf.minimum(tf.maximum(-1, dist),1)
    return dist






def try_sum(x1,x2):
    try:
        result = x1+x2
    # when sum(None, None)
    except TypeError:
        result = x1
    return result  
  
def get_models1(input):
    """Returns very simple CNN tf.keras.Model instance with N-1 shared layers for both tasks and single task-specific layer.
    """

    inp = tf.keras.layers.Input(shape = (1,input.shape[2]))
        
    out1=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,activation='linear',return_sequences=True), merge_mode='concat')(inp)
#    out1=tf.keras.layers.Bidirectional(LSTM(100,activation='linear',return_sequences=True), merge_mode='concat')(X)
    out2=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(50, activation='tanh'))(out1)
    out22=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20))(out2)
#    GRU(100,  activation='linear', return_sequences=True)(title_input)
    out222=tf.keras.layers.Dense(20, activation='tanh')(out22)

    
#    GRU(50, activation='tanh', return_sequences=False)(out1)
    out3=tf.keras.layers.Dropout(0.1)(out222)
    main= tf.keras.layers.Dense(1)(out3)
    
#    x = tf.keras.layers.Conv2D(32, 3,padding='same',activation='relu')(inp)
#    x = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
#    x = tf.keras.layers.Conv2D(64, 3,padding='same',activation='relu')(x)
#    x = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
#    x = tf.keras.layers.Flatten()(x)
#
#    main = tf.keras.layers.Dense(128, activation='relu')(x)
#    aux = tf.keras.layers.Dense(128, activation='relu')(x)
#    main = tf.keras.layers.Dense(2)(main)
#    aux = tf.keras.layers.Dense(2)(aux)

    return (tf.keras.Model(inputs = inp, outputs = main))#, tf.keras.Model(inputs = inp, outputs = aux))

def get_models(input):
    """Returns very simple CNN tf.keras.Model instance with N-1 shared layers for both tasks and single task-specific layer.
    """
#    inp = tf.keras.layers.Input(shape = (1,input.shape[1]))
    inp = tf.keras.layers.Input(shape = (input.shape[1],))
#    
#   
#    x = tf.keras.layers.Bidirectional(LSTM(100,return_sequences=True), merge_mode='concat')(inp)
#    x = tf.keras.layers.TimeDistributed(Dense(50, activation='tanh'))( x)
#    x = tf.keras.layers.Bidirectional(LSTM(20,return_sequences=False))(x)
##    x = tf.keras.layers.Dense(20, activation='tanh')(x)
#    x = tf.keras.layers.Dropout(0.1)(x)
    
#    inp = Input(shape=(1,input.shape[2]))
   
    x = tf.keras.layers.Dense(100, activation='tanh')(inp)
    main = tf.keras.layers.Dense(1, activation='linear')(x)
#    aux = tf.keras.layers.Dense(1 , activation='linear')(x)
#    main = tf.keras.layers.Dense(2)(main)
#    aux = tf.keras.layers.Dense(2)(aux)
#    return (tf.keras.Model(inputs = inp, outputs = main), tf.keras.Model(inputs = inp, outputs = aux))
    return (tf.keras.Model(inputs = inp, outputs = main))
####################################################################################################
#    title_input = Input(shape=(1,input.shape[2]))
#    out1=Bidirectional(LSTM(100,return_sequences=True), merge_mode='concat')(title_input)
#
#    out2=Dense(20, activation='tanh')(out1)
#
#    out3=Dropout(0.05)(out2)
#    main_output= Dense(1)(out3)   
#    model = Model(inputs=[title_input],outputs=[main_output])
#
#    return (model)
    

#Training parameters for a,b,c experiments    
epochs = 500 # number of epochs
steps = 1000 # numbers of steps per epoch
batch_size = 1
lr = 1e-6 # higher lr causes gradients to be unstable - loss doesn't decrease nicely
cos_sim_dict = {} # dictionary to save cosine similarities

#############################################################
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
for train_index, val_index in kf.split(dataX):
    iFold = iFold+1
#    train_x, test_x, train_y, test_y,= X[train_index], X[val_index], y[train_index], y[val_index] # 这里的X_train，y_train为第iFold个fold的训练集，X_val，y_val为validation set
    x_train, x_test, y_train, y_test, ytime_train, ytime_test, ystatus_train, ystatus_test, ystatus2_train, ystatus2_test =\
        dataX[train_index], dataX[val_index], y[train_index], y[val_index], ytime[train_index], ytime[val_index], ystatus[train_index],ystatus[val_index],\
                           ystatus2[train_index],ystatus2[val_index]
###########################################################################################
    train_ds_main0 = tf.data.Dataset.from_tensor_slices((x_train, y_train))#.shuffle(buffer_size=X_train.shape[0])
#    data_methylation_train_ds_main0 = tf.data.Dataset.from_tensor_slices((data_methylation_train, y_train))
#    train_ds_main = train_ds_main0.map(lambda x, y: (tf.to_float(x), tf.one_hot(y, 2))).shuffle(1)
    test_ds_main0 = tf.data.Dataset.from_tensor_slices((x_test, y_test))#.shuffle(buffer_size=X_test.shape[0])
#    test_ds_main = test_ds_main0.map(lambda x, y: (tf.to_float(x), tf.one_hot(y, 2))).shuffle(1)    
#    data_methylation_test_ds_main0 = tf.data.Dataset.from_tensor_slices((data_methylation_test, y_test))

#########################################################################################################################   
    
    
    
    input_dim =x_train.shape[2]
#    input_dim_mRNA =data_mRNA.shape[2]
#    ytrue_dim=y_train.shape[1]
#    input_dim_methylation =data_methylation.shape[2]
#    output_dim = x_train.shape[1]
    n1 = y_train.shape[0]
#    k_n = theano.shared(np.asarray(n,dtype=theano.config.floatX),borrow=True)
    k_n = theano.shared(n1,borrow=True)
    
    k_ytime_train = theano.shared(ytime_train,borrow=True)
    N = theano.shared(n1,borrow=True)
    R_matrix = np.zeros([n1, n1], dtype=int)
    R_matrix =theano.shared(R_matrix,borrow=True)
#    input_dim0 =theano.shared(input_dim,borrow=True)
# Build model structure
#    model=Sequential()
#    model.add(Bidirectional(LSTM(50,return_sequences=False,activation='linear'), input_shape=(1,input_dim)))
    ##############################################3    
##############################################3    
    
    Y_hazard0=y_train[:,0]
    Y_survival=y_train[:,1]

#    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_train.shape[0],1])
#    Y_hazard=Y_hazard1M[-1,:,-1]
    
    t0, H0 = unique_set(Y_hazard0) # t:unique time. H original index.
    
    actual_event_index = np.nonzero(Y_survival)[0]
    H0 = [list(set(h) & set(actual_event_index)) for h in H0]
    ordinal_n = np.asarray([len(h) for h in H0])
    Hj=sum(H0[0:],[])
    
    k_ordinal_H = theano.shared(np.asarray(Hj),borrow=True)
    k_ordinal_t = theano.shared(t0,borrow=True)
    k_ordinal_n = theano.shared(ordinal_n,borrow=True)
 ####################################################################################################
    
      
#############################################################################################################################    
    #Test algorythm by training
    model_main= get_models1( x_train)
    model_aux = get_models1(x_train)
#    model_main, model_aux = get_models()

    model_main.summary()
    model_aux.summary()
    # Synchronize initial weights
#    for (var_main, var_aux) in zip(model_main.variables, model_aux.variables):
#        var_aux.assign(var_main.value())
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

#    input_dim0 =theano.shared(input_dim,borrow=True)
# Build model structure
    # gene Only
#    gene_input = Input(name='gene_input', shape=(1,input_dim))
##    out1=Bidirectional(LSTM(55,activation='linear',return_sequences=True,kernel_initializer=glorot_uniform(),kernel_regularizer=l2(reg),activity_regularizer=l2(0.001)), merge_mode='concat')(title_input)
#    out_gene=Bidirectional(LSTM(100,return_sequences=True), merge_mode='concat')(gene_input)
##    out2=TimeDistributed(Dense(50, activation='tanh'))(out1)
##    out_gene=Bidirectional(LSTM(20))(out2)
#    
##    auxiliary_output = Dense(1, activation='linear', name='aux_output')(out_gene)  
##    GRU(100,  activation='linear', return_sequences=True)(title_input)
#     # clinic Only
#    clinic_input = Input(name='clinic_input', shape=(1,input_dim_methylation))
#    
#    out_clinic=Bidirectional(LSTM(100,activation='tanh',return_sequences=False), merge_mode='concat')(clinic_input )
#    auxiliary_output = Dense(1, activation='linear', name='aux_output')(out_clinic) 
##    out_clinic=Bidirectional(LSTM(100,return_sequences=False), merge_mode='concat')(clinic_input )
##    
#    out21=TimeDistributed(Dense(50, activation='tanh'))( out_gene)
#    out22=Bidirectional(LSTM(20,return_sequences=False))(out21)
##    model.add(TimeDistributed(Dense(50, activation='tanh')))
##    model.add(Bidirectional(LSTM(20)))
#    # combined with GRU output
##    input_ = Input(shape=(12,8))
#   
##    com = Concatenate(axis=1)([out_gene, out_clinic])
#   
#     
#    out222=Dense(20, activation='linear')(out22)
#
#    
##    GRU(50, activation='tanh', return_sequences=False)(out1)
#    out3=Dropout(0.1)(out22)
#    main_output= Dense(1,activation='linear',name='main_output')(out3)
##    main_output1=main_output[:,-1,:]
##    auxiliary_output1=auxiliary_output[:,-1,:]
#    model = Model(inputs=[gene_input,clinic_input],outputs=[main_output, auxiliary_output])
#    model_main = Model(inputs=gene_input,outputs=main_output)
#    model_aux = Model(inputs=clinic_input,outputs=auxiliary_output)
#    model.summary()
#    model_main.summary()
#    model_aux.summary()
#    # Synchronize initial weights
#    for (var_main, var_aux) in zip(model_main.variables, model_aux.variables):
#        var_aux.assign(var_main.value())
#    # optimizer
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)    
    
    
    batch_size=n1
    batch_size_M=x_train.shape[0]
    batch_size_test=x_test.shape[0]
    priv_ind = -2
#################################################################################################  
    loss00=[]
    for epoch in range(epochs):
    # Iterate over both main and aux task training samples
#        for (batch, ((images_main, labels_main), (images_aux, labels_aux))) in enumerate(
#            zip(train_ds_main.batch(batch_size), train_ds_aux.batch(batch_size))):
#    
#            # Break if we already did enough steps
#            if batch == steps:
#                break
        avg_loss = tf.Variable(0,dtype=tf.float32) 
        for (batch,train0) in enumerate(train_ds_main0.batch(batch_size, drop_remainder=True)):
        # Compute losses
            x_train0=train0[0]
#            
            Y_true=train0[1]
            Y_hazard_train=train0[1][:,0]
            Y_train=train0[1][:,1]
#            Y_survival_train1=tf.reshape(Y_survival_train, [-1, batch_size,1])
#            Y_survival_train2=Y_survival_train1[-1,:,:] 
            Y_train1=tf.reshape(Y_train, [-1, batch_size,1])
            Y_train2=Y_train1[-1,:,:]
            with tf.GradientTape() as tape_main:
                loss_main = neg_log_pl(Y_train2, model_main(tf.to_float(x_train0)))
            avg_loss = tf.math.add(avg_loss,loss_main)#fn(labels, model(images)))
        loss00.append(avg_loss.numpy())
        
        
        for (batch,train0) in enumerate(train_ds_main0.batch(batch_size_M, drop_remainder=True)):
            x_train0M=train0[0]
#            x_train1=tf.reshape(x_train0, [-1, 1,12])
            Y_trueM=train0[1]
            Y_hazard_trainM=train0[1][:,0]
            Y_trainM=train0[1][:,1]
#            Y_survival_train1=tf.reshape(Y_survival_train, [-1, batch_size,1])
#            Y_survival_train2=Y_survival_train1[-1,:,:] 
            Y_train1M=tf.reshape(Y_train, [-1, batch_size,1])
            Y_train2M=Y_train1[-1,:,:]
            
            with tf.GradientTape() as tape_aux:
#                loss_aux = neg_log_pl(Y_train2M, model_aux(tf.to_float(x_train0M)))
                loss_aux = Ordinal_Loss0(Y_trueM, model_aux(tf.to_float(x_train0M)))
#                loss_aux = LOSS_L2(Y_trueM, model_aux(tf.to_float(x_train0M)))
#                loss_aux = ordinal_loss_grad_test(Y_trueM, model_aux(tf.to_float(x_train0M)))
#                loss_aux = ordinal_loss(Y_hazard_trainM,Y_train2M, model_aux(tf.to_float(x_train0M)))
#                loss_aux =tf.py_function(func=ordinal_loss, inp=[Y_trueM, model_aux(tf.to_float(x_train0M))], Tout=tf.float32)
#                loss_aux =tf.py_function(func=ordinal_loss, inp=[Y_hazard_trainM,Y_train2M, model_aux(tf.to_float(x_train0M))], Tout=tf.float32)
        # compute gradients for shared and private parameters of both tasks
        grads_main = tape_main.gradient(loss_main, model_main.variables)
        grads_aux = tape_aux.gradient(loss_aux, model_aux.variables)
        
        optimizer.apply_gradients(
            zip(
                grads_main[priv_ind:], 
                model_main.variables[priv_ind:]))
                
        optimizer.apply_gradients(
            zip(
                grads_aux[priv_ind:], 
                model_aux.variables[priv_ind:]))
        if get_grad_cos_sim(grads_main, grads_aux, priv_ind)>=0:
            grad_sum = [x1+x2 for (x1, x2) in zip(grads_main[:priv_ind],grads_aux[:priv_ind])]
            optimizer.apply_gradients(
                zip(
                    grad_sum, 
                    model_main.variables[:priv_ind]))
            optimizer.apply_gradients(
                zip(
                    grad_sum, 
                    model_aux.variables[:priv_ind]))
#            combined_updates+=1
        else:
            optimizer.apply_gradients(
                zip(
                    grads_main[:priv_ind], 
                    model_main.variables[:priv_ind]))
            optimizer.apply_gradients(
                zip(
                    grads_aux[:priv_ind], 
                    model_aux.variables[:priv_ind]))

#        # compute cosine similarity for shared gradients of both tasks and decide how to update shared parameters
#        # Upgrading only main model will case aux model to update at the same time - they share layers and weights
#        if get_grad_cos_sim(grads_main, grads_aux, priv_ind)>=0:
#            
#            grad_sum = [try_sum(x1,x2) for (x1, x2) in zip(
#                grads_main,grads_aux)]
#            
#            ## main model with combined gradients
#            optimizer.apply_gradients(
#                zip(
#                    [g for i, g in enumerate(grad_sum) ],
#                    [v for i, v in enumerate(model_main.variables)]))
#            
#            ## aux model with combined gradients
#            optimizer.apply_gradients(
#                zip(
#                    [g for i, g in enumerate(grad_sum) ],
#                    [v for i, v in enumerate(model_aux.variables)]))
#        else:
#            
#            ## aux model with main gradients
#            optimizer.apply_gradients(
#                zip(
#                     [g for i, g in enumerate(grads_main) ],
#                    [v for i, v in enumerate(model_main.variables)]))
#            
#            ## aux model with aux gradients
#            optimizer.apply_gradients(
#                zip(
#                    [g for i, g in enumerate(grads_aux) ],
#                    [v for i, v in enumerate(model_aux.variables)]))
#        
#        # Apply task specific dense layer parameters
#        optimizer.apply_gradients(
#            zip(
#                [g for i, g in enumerate(grads_main) ],
#                    [v for i, v in enumerate(model_main.variables)]))
#        optimizer.apply_gradients(
#            zip(
#               [g for i, g in enumerate(grads_aux) ],
#                    [v for i, v in enumerate(model_aux.variables)]))
#
##    ## Compute average loss and accurancy for train andtest subset of main task samples
##    avg_acc, avg_loss = get_avg_acc_loss(x_train, model_main)
##    test_avg_acc, test_avg_loss = get_avg_acc_loss(x_test, model_main)
##
##    # Append avg cos sim
##    avg_acc_dict['Proposed Method'].append(test_avg_acc.numpy())
##
##    ## Print stats
##    print("Epoch:{} | loss={:.3f} acc={:.2f}% | val_loss={:.3f} val_acc={:.2f}%".\
##    format(epoch+1, avg_loss.numpy(), avg_acc.numpy()*100,test_avg_loss.numpy(), test_avg_acc.numpy()*100))
        avg_loss_test = tf.Variable(0,dtype=tf.float32)    
        for (batch,test0) in enumerate(test_ds_main0.batch(batch_size_test, drop_remainder=True)):
        # Compute losses
            x_test0=test0[0]
#            x_train1=tf.reshape(x_train0, [-1, 1,12])
            Y_hazard_test=test0[1][:,0]
            Y_test=test0[1][:,1]
#            Y_survival_train1=tf.reshape(Y_survival_train, [-1, batch_size,1])
#            Y_survival_train2=Y_survival_train1[-1,:,:] 
            Y_test1=tf.reshape(Y_test, [-1, batch_size_test,1])
            Y_test2=Y_test1[-1,:,:]
            avg_loss_test1 = neg_log_pl(Y_test2, model_main(tf.to_float(x_test0)))
            prediction1=model_main(tf.to_float(x_test0)).numpy()
            for (batch1,test1) in enumerate(test_ds_main0.batch(batch_size_test, drop_remainder=True)):
                x_test1=test1[0]
                Y_trueM=test1[1]
                Y_hazard_test=test0[1][:,0]
                Y_test=test1[1][:,1]
                Y_test1=tf.reshape(Y_test, [-1, batch_size_test,1])
                Y_test2=Y_test1[-1,:,:]
                
                avg_loss_test2 = LOSS_L2(Y_trueM, model_main(tf.to_float(x_test1)))
                prediction2=model_main(tf.to_float(x_test1)).numpy()
            avg_loss_test = tf.math.add(avg_loss_test,avg_loss_test1,avg_loss_test2)
            prediction=prediction1+prediction2
            
        print("Epoch:{} | train_loss={:.3f} | test_loss={:.3f}". format(epoch+1, avg_loss.numpy(),avg_loss_test.numpy()))
#        
    c_index2=c_index3( np.asarray(ytime_test),np.asarray(prediction), np.asarray(ystatus_test))
    
    print( c_index2)
    plt.plot(range(epoch+1), loss00)
    plt.show() 

#################################################################################################
##    embedding = Embedding(nb_features,word_size)
##    lstm_encoder = LSTM(encode_size)
##    inp = Input(shape=(1,input_dim,))
##    inp=Bidirectional(LSTM(55,activation='linear', return_sequences=False,\
##         kernel_initializer=glorot_uniform(),kernel_regularizer=l2(reg),activity_regularizer=l2(0.001)), input_shape=(1,input_dim), merge_mode='concat')
#    inp = Input(shape=(1,input_dim,))
#    x1=Bidirectional(LSTM(55,activation='linear',return_sequences=False,kernel_initializer=glorot_uniform(),kernel_regularizer=l2(reg),activity_regularizer=l2(0.001)), merge_mode='concat')(inp)
#
##    model.add(TimeDistributed(Dense(55)))
##    model.add(Bidirectional(LSTM(20)))
##    model.add(Dropout(0.005)) # important
##    model.add(Dropout(0.1))
##    model.add(BatchNormalization())
##    model.add(TimeDistributed(Dense(1, activation='linear')))
##    x2 = FMLayer(55, 10)(x1)
#    x3=(Dense(1))(x1) 
##    losslayer=Loss_Layer()
##    f_size = 2
##    f = Dense(f_size)(x1)
##   
##    losslayer= Loss_Layer(1)
#    output = Loss_Layer(1)(x3)    
##    ytrue_dim = Input(shape=(2,))
#    
##    output= Loss_Layer(f_size)(x2)
## Compile model
##    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, \
##        epsilon=None, decay=0.1), loss = neg_log_pl)
#    model=Model(inp, output)
#    model.compile(optimizer='adam',  loss={'main_output':neg_log_pl, 'aux_output':neg_log_pl})
#    
#    hist=model.fit({'gene_input':x_train,'clinic_input':data_methylation_train}, {'main_output': y_train, 'aux_output': y_train}, batch_size = n1, epochs =550)
#    
#    (predicted_main, predicted_aux)=model.predict({'gene_input': x_test,'clinic_input':data_methylation_test},verbose=1)
#    
##    prediction =predicted_main+0.5*predicted_aux c_index=0.7535211267605634
##    prediction =predicted_main+0.2*predicted_aux c_index=0.7591358964598401
#    prediction =predicted_main+0*predicted_aux
#    model.add(Bidirectional(LSTM(55,activation='linear', return_sequences=False,\
#        kernel_initializer=glorot_uniform(),kernel_regularizer=l2(0.0005),activity_regularizer=l2(0.001)), input_shape=(1,input_dim), merge_mode='concat'))
#    inp = Input(shape=(1,input_dim,))
#    x=LSTM(55,activation='linear')(inp)
#
#    x=Dense(1,activation='linear')(x)
#    
#     #将损失层加入网络
#
#    f_size = 1
#    inputs = Input(shape=(input_dim,))   #column
#    f = Dense(f_size)(x)
## a layer instance is callable on a tensor, and returns a tensor
#    losslayer=Loss_Layer(1)
#    predictions= losslayer(f)
#
#    model= Model(inp, x)
##    model.add(Loss_Layer()(xx))
#    model.compile(loss=neg_log_pl_1, optimizer='adam')
# Compile model
#    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, \
#        epsilon=None, decay=0.1), loss = neg_log_pl)
#    model.compile(optimizer='adam',  loss=neg_log_pl_1)
    
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
    
#    hist=model.fit(x_train, y_train, batch_size = n1, epochs =450)
#    hist=estimator.fit(x_train, y)
#    hist.history['loss']
    
    
#    prediction = model.predict(x_test)
#    prediction_train_median = model.predict(x_train)
    ypred.extend(prediction)
#    ypred_train.extend(prediction_train_median)
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
    
np.savetxt("brca_prediction1204_10fold_lstm_age_stage.csv", ypred, delimiter=",")
np.savetxt("brca_ytime_test1204_10fold_lstm_age_stage.csv", time_new, delimiter=",")
np.savetxt("brca_ystatus_test1204_10fold_lstm_age_stage.csv", status_new, delimiter=",")
np.savetxt("brca_ypred_train_median1204_10fold_lstm_age_stage.csv", ypred_train, delimiter=",")
c_indices_lstm = c_index3(np.asarray(time_new), np.asarray(ypred),np.asarray(status_new))
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
df = pd.read_csv("brca_prediction1204_10fold_lstm_age_stage.csv",header=None)    
month=np.asarray(pd.read_csv("brca_ytime_test1204_10fold_lstm_age_stage.csv",header=None)) 
status=np.asarray(pd.read_csv("brca_ystatus_test1204_10fold_lstm_age_stage.csv",header=None)) 


#df=ypred
#month=time_new
#status=status_new
#df1 = pd.read_csv("WAVE_preTest_scores.csv")    
#dataset_init = np.asarray(df1)    # if only 1 column
#dataX, dataY = create_interval_dataset(df, 1)    #这里的输入数据来源是csv文件

risk=np.asarray(df)
c_indices_lstm = c_index3(month, risk,status)
#np.savetxt("c_indices_nn827.txt", c_indices_mlp, delimiter=",")
np.save("c_indices_nn1204_10fold_lstm_age_stage.csv",c_indices_lstm) 
print(c_indices_lstm)
data_a=np.load('c_indices_nn1204_10fold_lstm_age_stage.csv.npy')
aa=0