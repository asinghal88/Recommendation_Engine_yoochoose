# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 04:37:21 2018

@author: abc
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from pandas.compat import StringIO
from pandas import DataFrame
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse, io
print('installed')

a=io.mmread('train1.mtx')
X_train=a.tocsr()
print('csr')
    
y_train=pd.read_csv('ytrain.csv',index_col=0,sep=',')
y_train.set_index('Target', inplace=True)
y_train=y_train.reset_index().values
y_train=y_train.astype(np.float)
y_train=y_train.reshape((9900007,))
print('start of model")
fm = pylibfm.FM(num_factors=3, num_iter=100, verbose=True, task="classification", initial_learning_rate=0.01, learning_rate_schedule="optimal")

fm.fit(X_train,y_train)
print('model ended')
#print('test')
#X_test=pd.read_table('xtest.csv',sep=',',index_col=0)
#def loadData(df):
#    data = []
#    #y = []
#    #users=set()
#    #items=set()
#    for i in range(len(df)):
#            data.append( {"category":str(df.iloc[i,0]),"items_in_category":str(df.iloc[i,1]),"Session_ID":str(df.iloc[i,2]),
#                          "categories_clicked":str(df.iloc[i,3]),"items_clicked":str(df.iloc[i,4]),"Item_ID":str(df.iloc[i,5]),
#                          "count":str(df.iloc[i,6]),"previous":str(df.iloc[i,7]),"next":str(df.iloc[i,8]),
#                          "day":str(df.iloc[i,9]),"month":str(df.iloc[i,10]),"hour":str(df.iloc[i,11]),
#                          "diff":str(df.iloc[i,12]),"position":str(df.iloc[i,13])})
#            #y.append(df2.iloc[i,:])
#           #users.add(user)
#            #items.add(movieid)
#            if i%100000==0:
#                print(i)
#    return (data, np.array(y))  
#X_test = v.transform(X_test)
#print('transformed_test')
#X_test=X_test.astype(np.float)
#y_test=y_test.astype(np.float)
#io.mmwrite("test.mtx", X_test)
#
#print('test_x')
#
