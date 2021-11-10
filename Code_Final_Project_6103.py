""" Introduction to Data Mining Project
    Group-4
    Names- Ricardo, Justin, Jeremiah, Osemekhian
"""
#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.tree import export_graphviz
import os
from pydotplus import graph_from_dot_data
import webbrowser

#--
pd.set_option("max_columns", None) # show all cols
pd.set_option('max_colwidth', None) # show full width of showing cols
pd.set_option("expand_frame_repr", False) #
#--

#Importing Dataset
##importing our datset from kaggle api chnage the username and kaggle key to your key if you dont want to use my key which is added to the gitup
os.environ['KAGGLE_USERNAME'] = 'koyanjo'
os.environ['KAGGLE_KEY'] = '33bfba07e0815efc297a1a4488dbe6a3'

from kaggle.api.kaggle_api_extended import KaggleApi

dataset = 'mohansacharya/graduate-admissions'
path = 'datasets/graduate-admissions'

api = KaggleApi()
api.authenticate()

api.dataset_download_files(dataset, path)

api.dataset_download_file(dataset, 'Admission_Predict.csv', path)
# use this if you the file is sql ->api.dataset_download_file(dataset, 'database.sqlite', path)

per = pd.read_csv("datasets/graduate-admissions/Admission_Predict.csv")
print(per.head())
print(per.describe())
per_new= per.copy() #DF copy

# DF cleaning, preprocessing
per_new.rename(columns={'Chance of Admit ':'Chance of Admit','LOR ': 'LOR'},inplace=True)
per_new['University Rating']=per_new['University Rating'].astype('category')
per_new = per_new.set_index('Serial No.')
per_new['GRE Groups'] = pd.cut(per_new['GRE Score'],4,labels= [1,2,3,4]) #Bining GRE into categories
per_new['TOEFL Groups'] = pd.cut(per_new['TOEFL Score'],4, labels = [1,2,3,4]) #Bining TOEFL into categories
per_new.head()
per_new.isna().any()

#Checking for Outliers with InterQuatile Range Detection for Int and Float

names = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA','Chance of Admit']

int_df = per_new.drop(['University Rating','Research','GRE Groups','TOEFL Groups'],axis =1)
n = 0
for e in range(1,7):
    item = int_df.iloc[:,n].values
    q25 = np.percentile(item,25)
    q50 = np.percentile(item,50)
    q75 = np.percentile(item,75)
    iqr = q75-q25
    cutoff = iqr * 3 #k=3
    lower,upper = q25 - cutoff, q75 + cutoff
    print(names[n], end = '')
    print(np.where(item>upper) or np.where(item<lower)) ## shows in which array the outlier appears
    n += 1

#Checking for Outliers for categoricals



#Visualization
sns.heatmap(per.corr(),annot=True,cmap='summer')
plt.title("Correlation on Admission Features");plt.show()
#---------
sns.jointplot(x=per_new['CGPA'],y=per_new['Chance of Admit']);plt.show()
sns.jointplot(x=per_new['CGPA'],y=per_new['Chance of Admit'],hue=per_new['Research']);plt.show()
sns.jointplot(x=per_new['CGPA'],y=per_new['Chance of Admit'],hue=per_new['University Rating']);plt.show()
sns.jointplot(x=per_new['GRE Score'],y=per_new['Chance of Admit'],hue=per_new['Research']);plt.show()
sns.jointplot(x=per_new['GRE Score'],y=per_new['Chance of Admit'],hue=per_new['University Rating']);plt.show()
sns.jointplot(x=per_new['TOEFL Score'],y=per_new['Chance of Admit'],hue=per_new['Research']);plt.show()
sns.jointplot(x=per_new['TOEFL Score'],y=per_new['Chance of Admit'],hue=per_new['University Rating']);plt.show()
sns.boxplot(x=per_new['Chance of Admit'],whis=np.inf); plt.show()
sns.boxplot(x=per_new['Research'],y=per_new['Chance of Admit'],hue=per_new['University Rating']); plt.show()
#--As CGPA and GRE increases, chance of admission increases also.

#Modeling
x,y = per_new.drop(['Chance of Admit','GRE Groups','TOEFL Groups'],axis=1), per_new['Chance of Admit']
# The algorithm model that is going to learn from our data to make predictions
# splitting the data 80:20
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)
# Fitting Decision Tree to Training set
regressor = DecisionTreeRegressor(max_depth=5,random_state=10) ##changing the depth because of our depth plot analysis
regressor.fit(X_train,y_train)


# Feature importance
feature_names = x.columns
feature_importance = pd.DataFrame(regressor.feature_importances_, index = feature_names)
print(feature_importance.sort_values)
## feature importance plot
features = list(feature_importance.index)
feature_importance.plot(kind='bar'); plt.show()
feature_importance.drop(['CGPA'],axis=0).plot(kind='bar'); plt.show()

#Prediction
pred=regressor.predict(X_test)
test_pred=pd.DataFrame({'Actual':y_test, 'Predicted':pred})
#Disbribution of predicted vs actual change of admittion
sns.kdeplot(data=test_pred,x='Actual',label ='Actual', color = 'olive')
sns.kdeplot(data= test_pred,x='Predicted',label ='Predicted', color = 'teal')
plt.xlabel("Chance of Admit")
plt.legend()
plt.show()


#checking the different settings on the decision tree, first depth
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

#plotting the difference range in depth and visualizing which is one is better
plt.plot(range(1,20),train_accuracy, marker = 'o' ,label = 'train_acc')##finding the appropiate depth
plt.plot(range(1,20),test_accuracy, marker = 'o' ,label = 'test_acc')
plt.xlabel('Depth of tree')
plt.ylabel('Performance')
plt.legend()
plt.show()
#---
plt.barh(frame['max_depth'],frame['train_acc'],color='red')
plt.barh(frame['max_depth'],-frame['test_acc'],color='green')
plt.ylabel('Depth of tree')
plt.title("TEST-------------- TRAIN")
plt.show()
#---

#Metric
from sklearn import metrics
score=regressor.score(X_test,y_test)
print(f'DecisionTree Regressor Score: {np.round(score,2)*100}%')
print('Mean Absolute Error:', np.round(metrics.mean_absolute_error(y_test, pred),4))
print('Mean Squared Error:', np.round(metrics.mean_squared_error(y_test, pred),4))
print('Root Mean Squared Error:', np.round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))

#Decision Tree's Tree
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
decision_tree = export_graphviz(regressor, out_file=None, feature_names = X_train.columns,filled = True , max_depth= 4)
# proffesor settings for the decision tree dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=data.iloc[:, 1:5].columns, out_file=None)
graph = graph_from_dot_data(decision_tree)
graph.write_pdf("decision_tree_gini3.pdf")
webbrowser.open_new(r'decision_tree_gini3.pdf')

# LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train) ## training on our dataset for linear regression
lr.get_params()

#Linear Regression Coefficients
lr_coef=pd.DataFrame({'features':x.columns,'coefficients':lr.coef_})

#Linear Regression Prediction
pred2 = lr.predict(X_test)
lr_pred = pd.DataFrame({'y_test':y_test,'y_pred':pred2})

sns.kdeplot(data=lr_pred,x='y_test',label ='Actual', color = 'olive')
sns.kdeplot(data= lr_pred,x='y_pred',label ='Predicted', color = 'teal')
sns.scatterplot(lr_pred['y_test'],lr_pred['y_pred'],color='blue')
plt.xlabel("Chance of Admit")
plt.legend()
plt.show()
score2=lr.score(X_test,y_test)
print(f'Linear Regression Model Score: {np.round(score2,2)}%')

# THIS DOESNT WORK!
#Residuals plot is used to analyze the variance of the error of the regressor.
#Residual Plot- Randomly scattered shows our linear model is good otherwise a non-linear model is preferred
# train=plt.scatter(lr.predict(y_train),(lr.predict(y_train)-y_train),color='green')
# test=plt.scatter(lr.predict(y_test),(lr.predict(y_test)-y_test),color='green')
# #plt.hlines(y=0,xmin=-10,xmax=10)
# plt.legend((train,test),('Training','Test'),loc='lower left')
# plt.title("Residual Plots")
#Metric


# DONT delete this
# sns.scatterplot(y_test,pred,color='green')
# plt.xlabel("Actual Chance of Admit")
# plt.ylabel("Predicted Chance of Admit")
# plt.title("Scatter Plot for Predicted vs Actual Chance of Admit [Decision Tree]")
# plt.show()