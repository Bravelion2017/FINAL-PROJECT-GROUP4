#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('Admission_Predict.csv')


# In[3]:


df.head(10)


# In[4]:


df.corr()


# In[5]:


df.index


# In[6]:


df.isnull().sum()


# In[7]:


df.min()


# In[8]:


df.max()


# In[9]:


df.mean()


# In[10]:


df.dtypes


# In[11]:


df.shape


# In[12]:


df['University Rating'] = df['University Rating'].astype('category') 
# this is a category column but since our dataset is not that big , time efficients is 
#not important also coding with a categorical sometimes its weird. I later changed it


# In[13]:


df.dtypes


# In[14]:


dummy = pd.get_dummies(df['University Rating'])


# In[15]:


dummy


# In[16]:


df = pd.concat([df,dummy], axis = 1)


# In[17]:


df.head()


# In[18]:


df = df.rename({1:'Rating 1', 2:'Rating 2', 3:'Rating 3',4:'Rating 4',5:'Rating 5'}, axis = 1)


# In[19]:


df.head(5)


# In[20]:


df.columns


# In[21]:


## Some columns has a space, renaming columns 


# In[22]:


df.rename({'Chance of Admit ':'Chance of Admit','LOR ': 'LOR'}, axis = 1, inplace = True )


# In[23]:


df.columns


# In[24]:


# Evaluatin good performance with 85> 

df = df.set_index('Serial No.')


# In[25]:


## setting index as serial no which is the same as index, is a redudant feature 


# In[41]:


df.head(20)


# In[ ]:


## creating a new column which recommends people to apply >0.70%

#df['American Dream'] = np.nan
#df['American'].apply(lambda x: 1 if df['Change of Admit'] > 0.75 else 0 )


# In[40]:


df['Chance of Admit']


# In[ ]:


#df['AmericanDream'] = df['Chance of Admit'].apply(lambda x: 1 if x > 0.75 else 0)


# In[ ]:


#df.head(20)


# In[ ]:


#df['AmericanDream'] = df[]apply(lambda x: (1 if x['Chance of Admit'] > 0.75 else 0), axis = 1)


# In[ ]:





# In[32]:


df['University Rating'] = df['University Rating'].astype('int')


# In[34]:


df.dtypes


# In[37]:


type(df)


# In[ ]:


## america dream, its most likely get accepted into a good university


# In[35]:


def myfunc(x,y):
    if x <= 0.75 and y > 2:
        return 1
    else:
        return 0 
## change of admit and >2 universtiy rating


# In[38]:


df['American Dream'] = df.apply(lambda x: myfunc(x['Chance of Admit'],x['University Rating']), axis = 1)


# In[39]:


df.head(10)


# In[45]:


df['CGPA'].unique()


# In[59]:


## diving the gre and toefl scores in 4 groups depending on the performance, just to increase features and have
# categorical
df['GRE Groups'] = pd.cut(df['GRE Score'],4,labels= [1,2,3,4])
df['TOEFL Groups'] = pd.cut(df['TOEFL Score'],4, labels = [1,2,3,4])


# In[60]:


df.head(20)


# In[49]:


df['GRE Groups'].value_counts()


# In[61]:


df['GRE Groups'].value_counts()


# In[55]:


df['University Rating'] = df['University Rating'].astype('category')


# In[56]:


df.dtypes


# In[ ]:


# check for An Imbalanced dataset is one where the number of instances of a class(es) 
#are significantly higher than another class(es), 
#thus leading to an imbalance and creating rarer class(es). check to choose the correct sampling


# In[ ]:


# We want to create an algorithm to make predictions if you 


# In[63]:


from sklearn.model_selection import train_test_split


# In[67]:


y = df['American Dream'] #yes and yes


# In[68]:


x = df.drop(['American Dream'], axis = 1)## all the features you wanna give to determine if you have american dream 


# In[88]:


x


# In[ ]:


# you dont need to normalize to use decision tree


# In[ ]:


# The algorithm model that is going to learn from our data to make predictions
# splitting the data 80:20 maybe perform validation scores to see tesplit size 


# In[137]:


X_train, X_valid, y_train, y_valid = train_test_split(x,y, test_size = 0.30)


# In[138]:


y_train.value_counts(normalize = True)


# In[139]:


y_valid.value_counts(normalize = True)


# In[140]:


X_valid.shape, y_valid.shape


# In[ ]:


#Things to remember however: the more training data you have, the better your model will be. 
#The more testing data you have, the less variance 
#you can expect in your results (ie. accuracy, false positive rate, etc.).


# In[141]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


# In[142]:


dt_model = DecisionTreeClassifier()


# In[143]:


dt_model.get_params()


# In[144]:


X_train


# In[145]:


dt_model.fit(X_train, y_train)


# In[146]:


predicitions = dt_model.predict(X_train)
predicitions


# In[147]:


dt_model.predict_proba(X_train) ## we dont have stopping criteria, our tree just grew


# In[153]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train,predicitions)


# In[157]:


from sklearn.metrics import precision_score
precision_score(y_train,predicitions)


# In[124]:


dt_model.score(X_train,y_train) #training score


# In[125]:


dt_model.score(X_valid,y_valid) #checking the validation score


# In[126]:


#y_pred = dt_model.predict(X_valid)


# In[136]:


#dt_model.predict_proba(X_valid)


# In[112]:


#from sklearn import metrics


# In[127]:


#print('Acurracy: ',metrics.accuracy_score(y_valid,y_pred))


# In[ ]:




