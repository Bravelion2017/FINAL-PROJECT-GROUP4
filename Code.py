#!/usr/bin/env python
# coding: utf-8

#Data Preprocessing

import pandas as pd
import numpy as np
df = pd.read_csv('Admission_Predict.csv')
df.head(10)
df.corr()
df.describe()

import seaborn as sns
import matplotlib.pyplot as plt

df2 = pd.read_csv('Admission_Predict.csv')
df2 = df2.iloc[:,1:]
df.head(5)
sns.heatmap(df.corr(), annot=True)  #This looks weird
plt.show()


df.index
df.isnull().sum()
df.min()
df.max()
df.mean()
df.dtypes
df.shape
df['University Rating'] = df['University Rating'].astype('category')

#Checking for duplicates
duplicate = df[df.duplicated()]
print(duplicate)

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

#df['American Dream'] = df.apply(lambda x: myfunc(x['Chance of Admit'],x['University Rating']), axis = 1)

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

df.head(5)
df.columns


## CREATING A TRAINING SET AND A TESTING SET

from sklearn.model_selection import train_test_split
y = df['Chance of Admit'] #yes and yes
## Different Rating doesnt make sense since its numerical
df.columns
df.drop(['Rating 1','Rating 2', 'Rating 3', 'Rating 4','Rating 5'], axis = 1 , inplace=True)
x = df.drop(['Chance of Admit'], axis = 1)## all the features you wanna give to determine if you have american dream

df.head()

pd.set_option("max_columns", None) # show all cols
pd.set_option('max_colwidth', None) # show full width of showing cols
pd.set_option("expand_frame_repr", False) #
# The algorithm model that is going to learn from our data to make predictions
# splitting the data 80:20 maybe perform validation scores to see testplit size
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.30)

y_train.value_counts(normalize = True)

y_test.value_counts(normalize = True)
X_test.shape, y_test.shape
X_train.shape,y_train.shape

plt.scatter(X_train,y_train, label ='Training Data', color ='r') # x and y must be the same size, fix this
plt.scatter(X_test,y_test, label = 'Testing Data', color = 'b')
plt.title('Splitting the Data')
plt.legend()
plt.show()

#Things to remember however: the more training data you have, the better your model will be.
#The more testing data you have, the less variance
#you can expect in your results (ie. accuracy, false positive rate, etc.).
# you dont need to normalize to use decision tree
## DECISION TREE MODEL
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

df.head(5)
## See which features are most important that the others.
regressor = DecisionTreeRegressor(random_state=10)

clf = regressor.fit(X_train,y_train) ## DO IT ONLY ON YOUR TRAINING SET
clf.get_params()
feature_names = x.columns
clf.feature_importances_
feature_importance = pd.DataFrame(clf.feature_importances_,index = feature_names)
print(feature_importance.sort_values)
df.head(10)

## feature importance plot
features = list(feature_importance[feature_importance[0]>0].index)
feature_importance.plot(kind='bar')
plt.show()



train_accuracy  = []
test_accuracy = []
##checking the different settings on the decision tree, first depth
for depth in range(1,20):
    regressor2 = DecisionTreeRegressor(max_depth=depth, random_state=10)
    regressor2.fit(X_train,y_train)
    train_accuracy.append(regressor2.score(X_train,y_train))
    test_accuracy.append(regressor2.score(X_test,y_test))

frame = pd.DataFrame({'max_depth': range(1,20),'train_acc':train_accuracy,'test_acc':test_accuracy})
frame.head(20)

dep = range(1,20)

#plotting the difference range in depth and visualizing which is one is better
plt.plot(dep,train_accuracy,marker = 'o',label = 'train_acc') ##finding the appropiate depth
plt.plot(dep,test_accuracy, marker = 'o' ,label = 'test_acc')
plt.xlabel('Depth of tree')
plt.ylabel('Performance')
plt.legend()
plt.show()

dt = DecisionTreeRegressor(max_depth=6,random_state=10)##changing the depth cause of our depth plot analysis
 ## accuracy on the test set is the same which is good
dt.fit(X_train,y_train)
dt.score(X_test,y_test)
dt.score(X_train,y_train)


from sklearn.metrics import accuracy_score, mean_squared_error ## mean squared error regression to check acc
X_test
pred = dt.predict(X_test)
pred
y_test
mean_squared_error(pred,y_test) ## the smallers the better



df.columns
import os
from sklearn import tree
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
import webbrowser

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
decision_tree = export_graphviz(dt, out_file=None, feature_names = X_train.columns,filled = True , max_depth= 4)
# proffesor settings for the decision tree dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=data.iloc[:, 1:5].columns, out_file=None)
graph = graph_from_dot_data(decision_tree)
graph.write_pdf("decision_tree_gini3.pdf")
webbrowser.open_new(r'decision_tree_gini3.pdf')

## LINEAR REGRESSION MODEL

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,y_train) ## training on our dataset for linear regression
lr.get_params()

lr.score(X_test,y_test)
#lr.score(X_train,y_train) doesnt make sense, only need accuracy against the testing dataset


X_test
pred = lr.predict(X_test)
mean_squared_error(pred,y_test)

pred.shape
type(pred)
y_test.shape
y_test.values
type(y_test)

linear_frame = pd.DataFrame()
linear_frame['pred'] = pred
linear_frame['y_test'] = y_test.values
linear_frame.head(5)
print(linear_frame)


X_train.shape
y_train.shape
#plt.scatter(X_train,y_train) ##shape of train are different
#plt.show()


## RANDOM FOREST

## KFOLD CROSS VALIDATION
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

cross_val_score(LinearRegression(),x,y)
cross_val_score(RandomForestRegressor(n_estimators=25),x,y)
cross_val_score(DecisionTreeRegressor(),x,y)



# ## BELOW DOESNT WORK, dont delete please

#final_clf.score(X_train,y_train) ##accuracy on the training set, which we will use to train our model, good score
#final_clf.score(X_test,y_test)  ## accuracy on the test set is the same which is good
# dt_model = DecisionTreeClassifier()
# dt_model.get_params()
# dt_model.fit(X_train, y_train) # error The Y variable must be the classification class
# dt_model.get_params()
# X_train
# feature_names = x.columns #taking the datafram column except chance of admit
# dt_model.feature_importances_

# predicitions = dt_model.predict(X_train)
# predicitions
# dt_model.predict_proba(X_train) ## we dont have stopping criteria, our tree just grew
# from sklearn.metrics import accuracy_score
# accuracy_score(y_train,predicitions)
# from sklearn.metrics import precision_score
# precision_score(y_train,predicitions)
#training score
# dt_model.score(X_train,y_train)
# #checking the validation score
# dt_model.score(X_test,y_test)
#y_pred = dt_model.predict(X_valid)
#dt_model.predict_proba(X_valid)
#from sklearn import metrics
#print('Acurracy: ',metrics.accuracy_score(y_valid,y_pred))
