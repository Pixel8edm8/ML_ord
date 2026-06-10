#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mRNA+methylation , epoch=500，c_index_lstm=0.611723022413926?
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
#from keras_contrib.layers import CRF
#import FMLayer
##############################
# load dataset
#kidtx = pd.read_csv('brca_Surv_data_stage_age.csv',usecols=['MEgreenyellow',	'MEgreen','MEturquoise'	,'MEmagenta','MEbrown',	'MEred',	'MEpink',	'MEblack',	'MEpurple',	'MEblue',	'MEyellow',	'MEgrey','erged_data5_stage','erged_data6_Age','V1','erged_data33'])
kidtx = pd.read_csv('brca_Surv_data_MM_0_age.csv')
#dataX = kidtx.drop(["V1", "erged_data33","erged_data5_stage"], axis = 1).values
dataX1 = kidtx.drop(["Unnamed: 0","methylation...17.","V1", "erged_data33"], axis = 1).values
y = np.transpose(np.array((kidtx["V1"], kidtx["erged_data33"]))) # V1=time; erged_data33=status


#dataX = kidtx.drop(["V1", "erged_data33","erged_data5_stage","erged_data6_Age"], axis = 1).values
#ytime =kidtx.loc[:, ["V1"]].values
#yevent=kidtx.loc[:, ["erged_data33"]].values#np.array((kidtx.loc[:,["erged_data33"]])).value
#stage=kidtx.loc[:, ["erged_data5_stage"]].values#np.array((kidtx.loc[:,["erged_data5_stage"]])).value
#age=kidtx.loc[:, ["erged_data6_Age"]].values #np.array((kidtx.loc[:,["erged_data6_Age"]])).value
#age=minmax_scale(age)  

#x = kidtx.drop(["V1", "erged_data33"], axis = 1).values
y = np.transpose(np.array((kidtx["V1"], kidtx["erged_data33"]))) # V1=time; erged_data33=status
#age=np.transpose(np.array((kidtx["erged_data6_Age"])))
#age=minmax_scale(age) 
#age=kidtx.loc[:, ["erged_data6_Age"]].values #np.array((kidtx.loc[:,["erged_data6_Age"]])).value
#age=minmax_scale(age)  
#x = torch.cat((dataX, age), 1)
dataX = np.asarray(dataX1)
dataX =minmax_scale(dataX ) 
data_methylation=dataX1[:,0:12]
data_mRNA=dataX1[:,12:39]
[ m,n] = dataX.shape
[ m1,n1] = data_methylation.shape
[ m2,n2] = data_mRNA.shape


#concat = np.zeros([m,n+1])
# 
#concat[:, : n] = dataX
#concat[:, n:] =age
#
#x = concat
##x = dataX
#x = np.asarray(x)
##[ m,n] = x.shape
#
#stage=kidtx.loc[:, ["erged_data5_stage"]].values #np.array((kidtx.loc[:,["erged_data6_Age"]])).value
#stage=minmax_scale(stage)  
#x = torch.cat((dataX, age), 1)

#[ m,n] = dataX.shape
#concat1 = np.zeros([m,n+1])
# 
#concat1[:, : n] = dataX
#concat1[:, n:] =stage
#
#x1=concat1
#
##age_stage = Concatenate(axis=1)([age,stage])
#concat2 = np.zeros([m,2])
#concat2[:, : 1] = age
#concat2[:, 1:] =stage
#age_stage=age
#
#[ m1,n1] = age_stage.shape
#age_stage=age_stage.reshape(m1,1,n1)


 
dataX = dataX.reshape(m,1,n)

data_methylation = data_methylation.reshape(m1,1,n1)
data_mRNA = data_mRNA.reshape(m2,1,n2)

ytime=np.transpose(np.array(kidtx["V1"])) # only V1=time;
ystatus= np.transpose(np.array(kidtx["erged_data33"])) #only erged_data33=status
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
 #####################################################################  
  ###########################################################################  
    
#    for i in range(time.shape[0] - 1, 0, -1):
#        # Going from smallest survival times to largest
#        if time[i] == time[i - 1]:
#            # Push risk to the later time (earlier in array position)
#            risk[i - 1] = risk[i - 1] + risk[i]
#            risk[i] = 0
            
            
            
            
######################################################################################################
 ######################################################################################           
    event = K.gather(y_true[:, 1], indices = sorting.indices)
    denom = K.cumsum(risk) #这个函数的功能是返回给定axis上的累计和
    terms = xbeta - K.log(denom)
    loglik = K.cast(event, dtype = terms.dtype) * terms   #cast将x的数据格式转化成dtype
    return -K.sum(loglik)

#numba  Numba 可以实现提升速度但又不需要改写部分代码为其他编程语言。
    
#from numba import vectorize, int64
# 
#@vectorize([int64(int64)], target='parallel')
    
#from numba import jit
#@jit(nopython=True)
#@jit
def my_func(array1):
    return array1

    
def add_rankLoss_numba(x):
   
#    n_sample =(x.shape[0]).value
#    R_matrix = tf.zeros([n_sample ,n_sample ],tf.int32)
#    for i in range(n_sample-1):
#        for j in range(N-1):
#            R_matrix[i,j] = x[j] >= x[i]
#    return (R_matrix)
#    with tf.Session() as sess:
#        print ( sess.run(x))
    n_sample =(x.shape[0]).value
#    R_matrix = tf.zeros([n_sample])
#    x2=np.zeros([n_sample])
#    x2=x
    x1 = tf.placeholder(tf.float32,[n_sample,None],name = 'array1')
    x2 = tf.py_function(my_func,[x1],tf.float32)

#    print(sess.run(x2))
    sess = tf.Session()
    feed_dict = {x1: x} 
    y2 = sess.run(x2, feed_dict = feed_dict)
    print(y2)
#    sorting = tf.nn.top_k(x[:, 0], k =int(k_n.get_value()))
#    R_matrix =K.gather(x[:, 0], indices = sorting.indices)
    R_matrix=np.zeros([n_sample])
     
    for i in range(n_sample - 1, 0, -1):
        # Going from smallest survival times to largest
        if tf.less(x[i], x[i-1]) is True:
            # Push risk to the later time (earlier in array position)
#            R_matrix[i].assign( 1,dtype=R_matrix.dtype)
            R_matrix[i]=1
        else:
#            R_matrix[i].assign(0,dtype=R_matrix.dtype)
            R_matrix[i]=0
#    for i in range(n_sample-1):
#        for j in range(n_sample-1):
#            R_matrix[i,j].assign( tf.cast(tf.less(x[i], x[j]),dtype=R_matrix.dtype))
#            print(i)
#    for i in range(n_sample-1):
#        if x[i]> x[i-1]:
#            R_matrix[i,j].assign( tf.cast(tf.less(x[i], x[j]),dtype=R_matrix.dtype))
#            print(i)
    return (R_matrix)

#from multiprocessing.dummy import Pool as ThreadPool
#    items = list()
#    pool = ThreadPool()
#    pool.map(process, items)
#    pool.close()
#    pool.join()
#a = tf.Variable(tf.zeros([3,3]))
#a[:,0:2].assign(1)
def risk_ab(tensor_a, tensor_b):
    risk_i,risk_i_1 = tf.py_func(compare_risk_i, [tensor_a, tensor_b], tf.float32)
    return risk_i,risk_i_1   

def compare_risk_i(a, b):
#    if a[0]==1.:
#        tile_b = np.tile(b, (4, 1))
#    else:
#        tile_b = b
#    return tile_b
    if a<=b:
        b = a+b
    else:
        a =0
    return a,b


def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
	n_sample =(x.shape[0]).value
	matrix_ones = tf.ones([n_sample, n_sample],tf.int32)
	indicator_matrix = tf.linalg.band_part(matrix_ones,-1,0) #下三角形

	return(indicator_matrix)
    
def neg_log_pl_1(y_true, y_pred):
    
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
class Loss_Layer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Loss_Layer, self).__init__(**kwargs)
        
    def neg_log_pl_1(self, y_true, y_pred):
 
        sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
        ytime = K.gather(y_true[:, 0], indices = sorting.indices)
        yevent = K.gather(y_true[:, 1], indices = sorting.indices)
        xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
        risk = K.exp(xbeta)
        
        n_observed = tf.reduce_sum(yevent,0)
        ytime_indicator = R_set(ytime)
        
        risk_set_sum =  K.sum(K.cast(ytime_indicator, dtype = y_pred.dtype)*(risk),axis=1)
        diff = xbeta - K.log(risk_set_sum)
        sum_diff_in_observed = K.cast(yevent, dtype = diff.dtype)*(diff)
        cost = (- K.sum(sum_diff_in_observed))
        return(cost)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)    

#    def build(self, input_shape):
#        # 添加可训练参数
#        self.kernel = self.add_weight(name='kernel',
#                                      shape=(input_shape[1], self.output_dim),
#                                      initializer='glorot_normal',
#                                      trainable=True)
#        self.bias = self.add_weight(name='bias',
#                                    shape=(self.output_dim,),
#                                    initializer='zeros',
#                                    trainable=True)
#        self.centers = self.add_weight(name='centers',
#                                       shape=(self.output_dim, input_shape[1]),
#                                       initializer='glorot_normal',
#                                       trainable=True)
    
    def call(self, inputs):
        self.inputs=inputs
#        y_true = inputs[0]
#        y_pred = inputs[1]
#        loss = self.neg_log_pl_1( y_true, y_pred)
#        self.add_loss(loss, inputs=inputs)
#        return K.dot(inputs, self.kernel) + self.bias
        return self.inputs
#        return K.dot(inputs, self.kernel)
    
def neg_log_pl_2(y_true, y_pred):
    
#    n_sample =(y_true.shape[0]).value
    n_sample =int(k_n.get_value())
    ytime = y_true[:, 0]
    yevent = y_true[:, 1]
    xbeta = y_pred[:, 0]
    risk = K.exp(xbeta)
    
    for i in range(n_sample - 1, 0, -1):
        # Going from smallest survival times to largest
        if ytime[i] == ytime[i - 1]:
            # Push risk to the later time (earlier in array position)
            risk[i - 1] = risk[i - 1] + risk[i]
            risk[i] = 0
 
 
    matrix_ones=np.ones([n_sample])
    ytime_indicator = np.tril(matrix_ones,-1) #下三角形
    
    risk_set_sum =  K.sum(K.cast(ytime_indicator, dtype = y_pred.dtype)*(risk),axis=1)
#    risk_set_sum =  K.sum(K.cast(K.exp(xbeta), dtype = ytime_indicator.dtype)*(ytime_indicator),axis=1,dtype=float32)
    
    diff = xbeta - K.log(risk_set_sum)
#    sum_diff_in_observed = K.cast(K.transpose(diff), dtype = yevent.dtype)*(yevent)
    sum_diff_in_observed = K.cast(yevent, dtype = diff.dtype)*(diff)
#    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    cost = (- K.sum(sum_diff_in_observed))
    
#    xent_loss = objectives.binary_crossentropy(x, xbeta)
#    z = 0
#    N=int(k_n.get_value())
#    for j in range(N):
#        tempz=0.000001
#        for i in range(j):
#            tempz = tempz + K.exp(y_pred[i,0])
#        z = z - K.log(tempz)
#    z = -z    
    return cost
    
#    train_ystatus = createSharedDataset(ystatus_train, cast_int=False)


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
for train_index, val_index in kf.split(dataX):
    iFold = iFold+1
#    train_x, test_x, train_y, test_y,= X[train_index], X[val_index], y[train_index], y[val_index] # 这里的X_train，y_train为第iFold个fold的训练集，X_val，y_val为validation set
    x_train, x_test, y_train, y_test, ytime_train, ytime_test, ystatus_train, ystatus_test,data_methylation_train,data_methylation_test,data_mRNA_train,data_mRNA_test =\
        dataX[train_index], dataX[val_index], y[train_index], y[val_index], ytime[train_index], ytime[val_index], ystatus[train_index],ystatus[val_index],\
        data_methylation[train_index],data_methylation[val_index],data_mRNA[train_index],data_mRNA[val_index]
    
    
    
    
    input_dim =x_train.shape[2]
    input_dim_mRNA =data_mRNA.shape[2]
#    ytrue_dim=y_train.shape[1]
    input_dim_methylation =data_methylation.shape[2]
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
    N = x_train.shape[0]
    dropout = 0.005
    batch_size = 128
    tau = 0.0871109017617 # obtained from BO
    lengthscale = 1e-2
    reg = lengthscale**2 * (1 - dropout) / (2. *n1 * tau)
    
      
#############################################################################################################################    
#    input_dim0 =theano.shared(input_dim,borrow=True)
# Build model structure
    # gene Only
    gene_input = Input(name='gene_input', shape=(1,input_dim_mRNA))
#    out1=Bidirectional(LSTM(55,activation='linear',return_sequences=True,kernel_initializer=glorot_uniform(),kernel_regularizer=l2(reg),activity_regularizer=l2(0.001)), merge_mode='concat')(title_input)
    out_gene=Bidirectional(LSTM(100,return_sequences=True), merge_mode='concat')(gene_input)
#    out2=TimeDistributed(Dense(50, activation='tanh'))(out1)
#    out_gene=Bidirectional(LSTM(20))(out2)
    
#    auxiliary_output = Dense(1, activation='linear', name='aux_output')(out_gene)  
#    GRU(100,  activation='linear', return_sequences=True)(title_input)
     # clinic Only
    clinic_input = Input(name='clinic_input', shape=(1,input_dim_methylation))
    
    out_clinic=Bidirectional(LSTM(100,activation='tanh',return_sequences=False), merge_mode='concat')(clinic_input )
    auxiliary_output = Dense(1, activation='linear', name='aux_output')(out_clinic) 
#    out_clinic=Bidirectional(LSTM(100,return_sequences=False), merge_mode='concat')(clinic_input )
#    
    out21=TimeDistributed(Dense(50, activation='tanh'))( out_gene)
    out22=Bidirectional(LSTM(20,return_sequences=False))(out21)
#    model.add(TimeDistributed(Dense(50, activation='tanh')))
#    model.add(Bidirectional(LSTM(20)))
    # combined with GRU output
#    input_ = Input(shape=(12,8))
   
#    com = Concatenate(axis=1)([out_gene, out_clinic])
   
     
    out222=Dense(20, activation='linear')(out22)

    
#    GRU(50, activation='tanh', return_sequences=False)(out1)
    out3=Dropout(0.1)(out22)
    main_output= Dense(1,activation='linear',name='main_output')(out3)
#    main_output1=main_output[:,-1,:]
#    auxiliary_output1=auxiliary_output[:,-1,:]
    model = Model(inputs=[gene_input,clinic_input],outputs=[main_output, auxiliary_output])
    model.summary()
    
    
    
    
    
   ############################################################################################  






#
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
    model.compile(optimizer='adam',  loss={'main_output':neg_log_pl, 'aux_output':neg_log_pl})
    
    hist=model.fit({'gene_input':data_mRNA_train,'clinic_input':data_methylation_train}, {'main_output': y_train, 'aux_output': y_train}, batch_size = n1, epochs =500)
    
    (predicted_main, predicted_aux)=model.predict({'gene_input': data_mRNA_test,'clinic_input':data_methylation_test},verbose=1)
    
#    prediction =predicted_main+0.5*predicted_aux c_index=0.7535211267605634
#    prediction =predicted_main+0.2*predicted_aux c_index=0.7591358964598401
    prediction =predicted_main+predicted_aux
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