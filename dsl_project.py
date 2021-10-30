# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 03:32:37 2018

@author: abc
"""

import pandas as pd
import numpy as np
from datetime import datetime
from decimal import *
from pandas import Series
from sklearn.cross_validation import train_test_split


clicks=pd.read_table('yoochoose-clicks.dat', header=None,sep=',',dtype=object)
clicks.columns=['Session_ID','Timestamp','Item_ID','category']
print('click')
buys=pd.read_table('yoochoose-buys.dat', header=None,sep=',',dtype=object)
buys.columns=['Session_ID','Timestamp','Item_ID','Price','Quantity']
print('buy')

buys=buys.iloc[:,[0,1,2,3]]
df2=clicks.iloc[:,[2,3]]

dict=Series(df2.category.values,index=df2.Item_ID).to_dict()
cat_from_itemid = lambda x: dict[x]
buys['category'] = buys.Item_ID.apply(cat_from_itemid)

del buys["Price"]

def functi(data):
  data['previous'] = data.groupby(['Session_ID'])['Item_ID'].shift(1)
  data['next'] = data.groupby(['Session_ID'])['Item_ID'].shift(-1)
  data['Timestamp']=pd.to_datetime(data['Timestamp'],format='%Y-%m-%dT%H:%M:%S.%fZ',errors='coerce')
  data['day']=data['Timestamp'].dt.strftime('%d')
  data['month']=data['Timestamp'].dt.strftime('%m')
  data['hour']=data['Timestamp'].dt.strftime('%H')
  data['epoch'] = pd.DatetimeIndex ( data['Timestamp'] ).astype(np.int64)/1000000
  data['diff'] =data.groupby('Session_ID')['Timestamp'].diff()
  click_number=data.groupby( ['Session_ID','Item_ID'] ).size().to_frame(name = 'count').reset_index()
  data= pd.merge(click_number,data, on=['Session_ID','Item_ID'])
  items_clicked=data.groupby( ['Session_ID'] ).size().to_frame(name = 'items_clicked').reset_index()
  data= pd.merge(items_clicked,data, on=['Session_ID'])
  data["position"]=data.groupby(['Session_ID']).cumcount()
  categories_clicked=data.groupby('Session_ID').category.nunique().to_frame(name = 'categories_clicked').reset_index()
  data= pd.merge(categories_clicked,data, on=['Session_ID'])
  items_in_category=data.groupby('category').Item_ID.nunique().to_frame(name = 'items_in_category').reset_index()
  data= pd.merge(items_in_category,data, on=['category'])
  return data
print('function')
clicks_final=functi(clicks)
print('clicks_final')
clicks_final.to_csv('clicks.csv')
del clicks_final['epoch']
del clicks_final['Timestamp']

buys_final=functi(buys)
buys_final.head()
buys_final.to_csv('buys.csv')

x=clicks_final["diff"]
x=x.dt.total_seconds()
clicks_final["diff"]=x
clicks_final.head()

buys_final["diff"]=buys_final["diff"].dt.total_seconds()
del buys_final["Timestamp"]
del buys_final["epoch"]


buys_final=buys_final.fillna(0)
clicks_final=clicks_final.fillna(0)

clicks_final["Target"]=0
buys_final["Target"]=1
train=pd.concat([clicks_final,buys_final],axis=0)
train_final = train.sample(frac=1).reset_index(drop=True)
train_final.head()
train_final.to_csv('train.csv')

data3=pd.read_table('train.csv',sep=',',dtype=object,index_col=0)

data3['category'] = data3['category'].astype('category')
data3['items_in_category'] = data3['items_in_category'].astype('category')
data3['Session_ID'] = data3['Session_ID'].astype('category')
data3['categories_clicked'] = data3['categories_clicked'].astype('category')
data3['items_clicked'] = data3['items_clicked'].astype('category')
data3['Item_ID'] = data3['Item_ID'].astype('category')
data3['previous'] = data3['previous'].astype('category')
data3['next'] = data3['next'].astype('category')
data3['day'] = data3['day'].astype('category')
data3['month'] = data3['month'].astype('category')
data3['position'] = data3['position'].astype('category')
data3['count'] = data3['count'].astype('category')
data3['hour'] = data3['hour'].astype('category')
data3['diff'] = data3['diff'].astype('float')
data3['Target'] = data3['Target'].astype('category')


X = data3.iloc[:,0:14]
y = data3.iloc[:,14:15]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.01)

def loadData(df,df2):
    data = []
    y = []
    #users=set()
    #items=set()
    for i in range(len(df)):
            data.append( {"category":str(df.iloc[i,0]),"items_in_category":str(df.iloc[i,1]),"Session_ID":str(df.iloc[i,2]),
                          "categories_clicked":str(df.iloc[i,3]),"items_clicked":str(df.iloc[i,4]),"Item_ID":str(df.iloc[i,5]),
                          "count":str(df.iloc[i,6]),"previous":str(df.iloc[i,7]),"next":str(df.iloc[i,8]),
                          "day":str(df.iloc[i,9]),"month":str(df.iloc[i,10]),"hour":str(df.iloc[i,11]),
                          "diff":str(df.iloc[i,12]),"position":str(df.iloc[i,13])})
            y.append(df2.iloc[i,:])
           #users.add(user)
            #items.add(movieid)
            if i%100000==0:
                print(i)
    return (data, np.array(y))   

X_train,Y_train=loadData(xtrain,ytrain)


import gc
gc.collect()
