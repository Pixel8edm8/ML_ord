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

loss500_ordCOX =  np.asarray(pd.read_csv("hist.history['loss1000_ml_ordCOX20201220']_without_penalty_term.csv",header=None))  
loss500_M_m = np.asarray(pd.read_csv("brca_methylation_mRNA_without_penalty_term_only main loss20201223.csv",header=None))
loss500_Methylation= np.asarray(pd.read_csv("brca_only_methylation_without_penalty_term_only main loss20201223.csv",header=None))
loss500_mRNA= np.asarray(pd.read_csv("brca_only_mRNA_without_penalty_term_only main loss20201223.csv",header=None))

  

epoch=np.arange(1000)
#plt.gca().set_color_cycle(['red','green','blue','black'])
#plt.plot(epoch,loss500_ordCOX)

plt.plot(epoch,loss500_Methylation[0:1000],color='green' )
plt.plot(epoch,loss500_mRNA[0:1000] ,color='purple')
plt.plot(epoch,loss500_M_m[0:1000],color='darkorange' )
plt.title('Loss Curves')
plt.ylabel('train-loss')
plt.xlabel('epoch')
plt.legend(['${G_meth}$','${G_mRNA}$','${G_mRNA-meth}$'], loc='upper right')
plt.savefig('Loss Curves123.jpg', dpi=300) #指定分辨率保存
plt.show()
#plt.savefig('Loss Curves.jpg', dpi=300) #指定分辨率保存
#a/${m_2}$ 
