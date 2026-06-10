# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 23:49:30 2020

@author: liugu
"""


import datetime
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt

loss500_ordCOX_with = np.asarray(pd.read_csv("hist.history['loss1000_ml_ordCOX20201220']_with_penalty_term0.csv",header=None))  
loss500_ordCOX_without = np.asarray(pd.read_csv("hist.history['loss1000_ml_ordCOX20201220']_without_penalty_term0.csv",header=None))
#loss500_Methylation= np.asarray(pd.read_csv("hist.history['loss500_Mythlation'].csv",header=None))
#loss500_mRNA= np.asarray(pd.read_csv("hist.history['loss500_mRNA'].csv",header=None))
loss500_ordCOX2=[]
#for i in range(2000):
#    if i>500:
#        loss500_ordCOX11=loss500_ordCOX1[500]-i/100+(loss500_ordCOX1[i]-loss500_ordCOX1[i-1])
#        loss500_ordCOX2.extend(loss500_ordCOX11)
#    if i<=500:
#        loss500_ordCOX2.extend(loss500_ordCOX1[i])
#
#for i in range(480,520):
#    loss500_ordCOX2[i]=loss500_ordCOX1[i]+i/100
epoch=np.arange(1000)
#plt.gca().set_color_cycle(['red','green','blue','black'])
plt.plot(epoch,loss500_ordCOX_without[0:1000],color='darkorange' )
plt.plot(epoch,loss500_ordCOX_with[0:1000],color='darkblue')

#plt.plot(epoch,loss500_Methylation )
#plt.plot(epoch,loss500_mRNA )
plt.title('Training Loss Curves with/without penalty term')
plt.ylabel('training-loss')
plt.xlabel('epoch')
plt.legend(['Training Loss without penalty term','Training Loss with penalty term'], loc='upper right')
plt.savefig('Training Loss Curves with and without penalty term.jpg', dpi=300) #指定分辨率保存
plt.show()
#plt.savefig('Loss Curves.jpg', dpi=300) #指定分辨率保存
#a/${m_2}$ 
