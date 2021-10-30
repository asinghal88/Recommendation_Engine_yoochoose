
# coding: utf-8

# In[ ]:

#!source activate ManaEnv
#!conda install -c conda-forge textblob -n ManaEnv -y


# In[1]:

import pandas as pd
import gzip
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords


# In[2]:

import numpy as np
from pyfm import pylibfm
from scipy.sparse import csr_matrix


# In[3]:

import nltk
import time
import math


# In[4]:

from textblob import TextBlob
import re,string
from string import digits
regex = re.compile('[%s]' % re.escape(string.punctuation))
remove_digits = str.maketrans('', '', digits)
#stop_words = set(stopwords.words('english'))
from nltk.stem.wordnet import WordNetLemmatizer 
lmtzr = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


# In[5]:

def spell_correct_using_blob(s):
    tb = TextBlob(s)
    return tb.correct()


# In[6]:

#stop_words = set(stopwords.words('english'))
noise_list = set(stopwords.words('english'))
def _remove_noise(input_text):
    words = input_text.split() 
    noise_free_words = [word for word in words if word not in noise_list] ### remove stop words
    noise_free_lemmed_words = [lmtzr.lemmatize(word,'v') for word in noise_free_words] ### stemming
    noise_free_text = " ".join(noise_free_lemmed_words) 
    return noise_free_text


# In[7]:

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


# In[ ]:

#df = getDF('reviews_Baby_5.json.gz')
df = getDF('reviews_Electronics_5.json.gz')


# In[ ]:

df.head()


# In[ ]:


text_vec = df['reviewText']
summary_vec = df['summary']
y = df['overall']


# In[ ]:

df['overall'].value_counts()
#y = df['overall'][0:1000]
#y = np.repeat(1.0,X.shape[0])


# In[8]:

##########
### Remove Punctuation
### Remove Numbers
### Convert text to lower case
### remove extra whitespace
### spell correction --- new
### remove stop words
### 

def basic_preprocessing_text(text_vec):
    tic = time.time()
    text_vec = [regex.sub('', s) for s in text_vec] ### 1
    print(time.time()-tic)
    text_vec = [s.translate(remove_digits) for s in text_vec] ### 2
    print(time.time()-tic)
    text_vec = [s.lower() for s in text_vec] ### 3
    print(time.time()-tic)
    text_vec = [' '.join(s.split()) for s in text_vec] ### 4
    print(time.time()-tic)
    #text_vec = [spell_correct_using_blob(s) for s in text_vec] ### 5
    text_vec = [_remove_noise(s) for s in text_vec] ### 5
    print(time.time()-tic)
    return text_vec


# In[9]:

#X = vectorizer.fit_transform(tv) 
from sklearn.feature_extraction import DictVectorizer


# In[ ]:

#print(X.toarray())
#print(vectorizer.get_feature_names())


# In[10]:

def balanced_subsample_pandas(x,y,subsample_size=1.0,min_mult = 1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = round(min_elems*min_mult)
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            this_xs = this_xs.reindex(np.random.permutation(this_xs.index))
        elems = min(use_elems,len(this_xs))
        print(elems,len(this_xs),use_elems)
        x_ = this_xs[:elems]
        y_ = np.empty(elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = pd.concat(xs)
    ys = pd.Series(data=np.concatenate(ys),name='target')

    return xs,ys


# In[ ]:




# In[ ]:

tv_cleaned = basic_preprocessing_text(text_vec)
#s_cleaned = basic_preprocessing_text(summary_vec)


# In[ ]:

#backup = pd.HDFStore('backup.h5')
#pd.to_csv('')
#backup['tv_cleaned'] = tv_cleaned

df2 = df.iloc[:,[0,2,5]]
df2['reviewText'] = tv_cleaned
df2['summary'] = s_cleaned


# In[ ]:




# In[ ]:

#df2.to_csv('cleaned_reviews.csv')


# In[267]:

df2 = pd.read_csv('cleaned_reviews.csv')


# In[ ]:

freq = pd.Series(' '.join(tv_cleaned).split()).value_counts()


# In[ ]:

(freq>2).value_counts()


# In[268]:

df3,ynew = balanced_subsample_pandas(df2,df2["overall"],1,1.2)


# In[290]:

#df3 = pd.read_csv('df3.csv')
df3.head()


# In[312]:

df3.iloc[50:52,3]
#df3 = df3.fillna('NA')


# In[311]:

df3['TotalText'] = [' '.join(x) for x in zip(df3.iloc[:,3],df3.iloc[:,2])]


# In[15]:

print([X_train.head()],y_train.head())


# In[313]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df3["TotalText"], df3["overall"], test_size=0.2, random_state=42)


# In[314]:

from sklearn.preprocessing import normalize


# In[315]:

def fit_fm_model_data_prep(tv):   
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tv) 
    X = X.astype(np.float)
    return X,vectorizer


# In[316]:

X,v1 = fit_fm_model_data_prep(X_train[0:100000])


# In[317]:

B = csr_matrix(X, copy=True)
B[B>0] = 1


# In[318]:

B


# In[365]:

MIN_VAL_ALLOWED = 1

z = np.squeeze(np.asarray(B.sum(axis=0) > MIN_VAL_ALLOWED)) #z is the non-sparse terms 


# In[366]:

X2 = B[:,z]


# In[367]:

X2


# In[175]:

#normalize(X2).toarray()


# In[363]:

fm2 = pylibfm.FM(num_factors = 4, num_iter=20, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")

fm2.fit(normalize(X2),y_train[0:100000])


# In[323]:

def my_predict_fm(fm,nt,v,normal):
    #nt = basic_preprocessing_text(nt)
    vector = v.transform(nt)
    #
    if (normal):
        preds = fm.predict(normalize(vector.astype(np.float)[:,z]))
    else:
        preds = fm.predict(vector.astype(np.float)[:,z])    
    return preds
    


# In[324]:

def my_predict_fm2(fm,nt,v,normal):
    nt = basic_preprocessing_text(nt)
    vector = v.transform(nt)
    #
    if (normal):
        preds = fm.predict(normalize(vector.astype(np.float)[:,z]))
    else:
        preds = fm.predict(vector.astype(np.float)[:,z])    
    return preds


# In[368]:

def my_predict_fm3(fm,nt,v,normal):
    nt = basic_preprocessing_text(nt)
    vector = v.transform(nt)
    vector[vector>0] = 1
    #
    if (normal):
        preds = fm.predict(normalize(vector.astype(np.float)[:,z]))
    else:
        preds = fm.predict(vector.astype(np.float)[:,z])    
    return preds


# In[347]:

text = ["I love this thing. worth every penny get sme rioght to my destination. only issue is that sometimes it seems to take a little longer to find itself if i make a stop."]


# In[378]:

text = ["Save money and buy this one. Works fine for the past 3 months. Do not buy the expensive Oppo and other products. "]


# In[379]:

my_predict_fm2(fm,text,v1,True)


# In[380]:

my_predict_fm3(fm2,text,v1,True)


# In[382]:

basic_preprocessing_text(['These are awesome and make my phone look so stylish! I have only used one so far and have had it on for almost a year! CAN YOU BELIEVE THAT! ONE YEAR!! Great quality! '])


# In[390]:

ax = X_test
ay = y_test
ax = X_train#[0:1000]
ay = y_train#[0:1000]

nor = True
p = my_predict_fm(fm,ax,v1,nor)


# In[206]:

ay[20:30],p[20:30]


# In[336]:

ay[220:230],p[220:230]


# In[391]:

p2 = np.round(p)


# In[338]:

pd.Series(p2).value_counts()


# In[387]:

pd.Series(p2).value_counts()


# In[393]:

from sklearn.metrics import mean_squared_error
print("FM RMSE: %.4f" % mean_squared_error(ay,p))


# In[392]:

from sklearn.metrics import accuracy_score
accuracy_score(ay,p2)


# In[402]:

def acc1(y_true, y_pred):
    flag = np.abs(y_true - y_pred)<=1
    return np.mean(flag)


# In[403]:

acc1(ay,p)


# In[ ]:


text2 = [text_vec[20]]
vector = v2.transform(text2)
print(vector.toarray())


# In[ ]:

nt = basic_preprocessing_text(text_vec[400:403])


# In[ ]:

my_predict_fm(fm,text_vec[400:407],v1)


# In[ ]:

y[400:407]


# In[ ]:

y.value_counts()


# In[ ]:

def balanced_subsample_pandas(x,y,subsample_size=1.0,min_mult = 1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = round(min_elems*min_mult)
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            this_xs = this_xs.reindex(np.random.permutation(this_xs.index))
        elems = min(use_elems,len(this_xs))
        print(elems,len(this_xs),use_elems)
        x_ = this_xs[:elems]
        y_ = np.empty(elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = pd.concat(xs)
    ys = pd.Series(data=np.concatenate(ys),name='target')

    return xs,ys


# In[ ]:

xnew,ynew = balanced_subsample_pandas(pd.Series(tv_cleaned),y,1,2)


# In[ ]:

xnew = list(xnew)


# In[ ]:

xnew


# In[ ]:

def remove_most_frequent_words(head,tail,vec):  
    freq = pd.Series(' '.join(vec).split()).value_counts()
    topwords = list(freq.index[:math.ceil(len(freq)*head/100)])
    bottomwords = list(freq.index[-math.ceil(len(freq)*tail/100):])
    totalwords = topwords+bottomwords
    #print (totalwords)
    vec = pd.Series(vec)
    vec2 = vec.apply(lambda x: " ".join(x for x in x.split() if x not in totalwords))
    return vec2,topwords,bottomwords

