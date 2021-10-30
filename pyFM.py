# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:47:14 2018

@author: abc
"""

!pip install git+https://github.com/coreylynch/pyFM
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from pandas.compat import StringIO
from pandas import DataFrame
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer


data= pd.read_csv('yoochoose-clicks.dat',header=None,sep=',',dtype=object)
data.columns=["user_id","timestamp","item_id","category_id"]
data["target"]=0
data["user_id"]=data["user_id"].astype("int64")
pos=data.sort_values("user_id")
pos=pos.iloc[0:2284398,[0,2,4]]
pos.head()


data_neg = pd.read_csv('yoochoose-buys.dat')
data_neg.columns=["user_id","timestamp","item_id","price","quantity"]
data_neg["target"]=1
data_neg["user_id"]=data_neg["user_id"].astype("int64")
neg=data_neg.sort_values("user_id")
neg=neg.iloc[0:75001,[0,2,5]]
neg.head()



train=pd.concat([pos,neg],axis=0)
train["user_id"]=train["user_id"].astype("int64")
train["item_id"]=train["item_id"].astype("int64")
train["target"]=train["target"].astype("int64")


def loadData(df):
    data = []
    y = []
    #users=set()
    #items=set()
    for i in range(len(df)):
            data.append({ "user_id": str(df.iloc[i,0]), "item_id": str(df.iloc[i,1])})
            y.append(df.iloc[i,2])
           #users.add(user)
            #items.add(movieid)
            if i%10000==0:
                print(i)
    return (data, np.array(y))





from scipy import sparse, io

(train_data, y_train) = loadData(train)
(test_data, y_test) = loadData("ua.test")
v=DictVectorizer()
X_train = v.fit_transform(train_data)
X_train=X_train.astype(np.float)
y_train=y_train.astype(np.float)
io.mmwrite("train.mtx", X_train)
np.savetxt('y_train.txt',y_train,delimiter='')
a=io.mmread('train.mtx')
X_train=a.tocsr()
y_train=pd.read_csv('y_train.txt',index_col=0)
y_train=y_train.reset_index().values
y_train=y_train.reshape((9900007,))
#X_train=normalize(X_train)


#X_test = v.transform(test_data)

# Build and train a Factorization Machine
fm = pylibfm.FM(num_factors=3, num_iter=100, verbose=True, task="classification", initial_learning_rate=0.01, learning_rate_schedule="optimal")

fm.fit(X_train,y_train)
