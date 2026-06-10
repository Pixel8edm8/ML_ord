# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:50:44 2019

@author: Administrator
"""

#import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
df = pd.read_csv("brca_ypred_train_median829_12.csv",header=None)    

#month=np.asarray(pd.read_csv("brca_ytime_test828_12.csv",header=None)) 
#status=np.asarray(pd.read_csv("brca_ystatus_test828_12.csv",header=None)) 

x = np.asarray(df)
s1=np.shape(df[0])
print(s1)
xpred=[]
for i in range(0,779):
    sum1=0
    for i1 in range(0,778):
        x1=x[i*778+i1]
        sum1=sum1+x1
    
    sum1=np.asarray(sum1)    
    xpred.extend(sum1)

np.savetxt("brca_pred778_779_829_12.csv", xpred, delimiter=",")
 
aa=0       