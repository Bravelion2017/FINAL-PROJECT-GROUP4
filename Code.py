#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
df = pd.read_csv('Admission_Predict.csv')
df.head(10)
df.corr()
df.index
df.isnull().sum()
df.min()
df.max()
df.mean()
df.dtypes
df.shape
df['University Rating'] = df['University Rating'].astype('category')
# this is a category column but since our dataset is not that big , time efficients is 
#not important also coding with a categorical sometimes its weird. I later changed it
df.dtypes
dummy = pd.get_dummies(df['University Rating'])
dummy
df = pd.concat([df,dummy], axis = 1)
df.head()
df = df.rename({1:'Rating 1', 2:'Rating 2', 3:'Rating 3',4:'Rating 4',5:'Rating 5'}, axis = 1)
df.head(5)
df.columns
## Some columns has a space, renaming columns
df.rename({'Chance of Admit ':'Chance of Admit','LOR ': 'LOR'}, axis = 1, inplace = True )
df.columns
# Evaluatin good performance with 85>
df = df.set_index('Serial No.')
df.head(20)

df['Chance of Admit']

df['University Rating'] = df['University Rating'].astype('int')

df.dtypes

type(df)

## america dream, its most likely get accepted into a good university

def myfunc(x,y):
    if x <= 0.75 and y > 2:
        return 1
    else:
        return 0 
## change of admit and >2 universtiy rating

df['American Dream'] = df.apply(lambda x: myfunc(x['Chance of Admit'],x['University Rating']), axis = 1)

df.head(10)

df['CGPA'].unique()

## diving the gre and toefl scores in 4 groups depending on the performance, just to increase features and have
# categorical
df['GRE Groups'] = pd.cut(df['GRE Score'],4,labels= [1,2,3,4])
df['TOEFL Groups'] = pd.cut(df['TOEFL Score'],4, labels = [1,2,3,4])

df.head(20)

df['GRE Groups'].value_counts()

df['GRE Groups'].value_counts()


df['University Rating'] = df['University Rating'].astype('category')

df.dtypes

# check for An Imbalanced dataset is one where the number of instances of a class(es) 
#are significantly higher than another class(es), 
#thus leading to an imbalance and creating rarer class(es). check to choose the correct sampling

# We want to create an algorithm to make predictions if you 

from sklearn.model_selection import train_test_split

y = df['American Dream'] #yes and yes

x = df.drop(['American Dream'], axis = 1)## all the features you wanna give to determine if you have american dream 

x

# you dont need to normalize to use decision tree
# The algorithm model that is going to learn from our data to make predictions
# splitting the data 80:20 maybe perform validation scores to see tesplit size 

X_train, X_valid, y_train, y_valid = train_test_split(x,y, test_size = 0.30)

y_train.value_counts(normalize = True)

y_valid.value_counts(normalize = True)


X_valid.shape, y_valid.shape

#Things to remember however: the more training data you have, the better your model will be. 
#The more testing data you have, the less variance 
#you can expect in your results (ie. accuracy, false positive rate, etc.).

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeClassifier()

dt_model.get_params()

X_train

dt_model.fit(X_train, y_train)

predicitions = dt_model.predict(X_train)
predicitions

dt_model.predict_proba(X_train) ## we dont have stopping criteria, our tree just grew

from sklearn.metrics import accuracy_score
accuracy_score(y_train,predicitions)

from sklearn.metrics import precision_score
precision_score(y_train,predicitions)

dt_model.score(X_train,y_train) #training score

dt_model.score(X_valid,y_valid) #checking the validation score

#y_pred = dt_model.predict(X_valid)

#dt_model.predict_proba(X_valid)

#from sklearn import metrics

#print('Acurracy: ',metrics.accuracy_score(y_valid,y_pred))

