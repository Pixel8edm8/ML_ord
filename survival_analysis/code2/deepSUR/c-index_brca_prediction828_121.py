#import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
# Main slim library
#from lib_1.contour import *
#from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_dilation
#from tensorflow.contrib import slim
#from nets import inception_v4
#
#from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
#from skimage.transform import resize
#from imgaug import augmenters as iaa
#from skimage import exposure
#from datasets import dataset_utils
#from sklearn.utils import shuffle
#
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

np.random.seed(13)



## C_index metric function

def c_index3(month,risk, status):

    c_index = concordance_index(np.reshape(month, -1), -np.reshape(risk, -1), np.reshape(status, -1))

    return c_index

df = pd.read_csv("brca_prediction828_121.csv",header=None)    
#dataset_init = np.asarray(df)    # if only 1 column
#dataX, dataY = create_interval_dataset(df, 1)    #这里的输入数据来源是csv文件
month=np.asarray(pd.read_csv("brca_ytime_test828_121.csv",header=None)) 
status=np.asarray(pd.read_csv("brca_ystatus_test828_121.csv",header=None)) 


#df1 = pd.read_csv("WAVE_preTest_scores.csv")    
#dataset_init = np.asarray(df1)    # if only 1 column
#dataX, dataY = create_interval_dataset(df, 1)    #这里的输入数据来源是csv文件

risk=np.asarray(df)
c_indices_mlp = c_index3(month, risk,status)
#np.savetxt("c_indices_nn827.txt", c_indices_mlp, delimiter=",")
np.save("c_indices_nn828_121.csv",c_indices_mlp) 
print(c_indices_mlp)
data_a=np.load('c_indices_nn828_121.csv.npy')

#fileObject = open('c_indices_nn.txt', 'w') 
#data=str(c_indices_mlp)
#fileObject.write(data) 
#fileObject.write('\n') 
aa=0
