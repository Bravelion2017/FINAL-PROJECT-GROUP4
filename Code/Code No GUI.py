""" Introduction to Data Mining Project
    Group-4
    Names- Ricardo, Junran, Jeremiah, Osemekhian
"""
# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import export_graphviz
import os
from pydotplus import graph_from_dot_data
from sklearn.ensemble import VotingRegressor
import webbrowser

# --
pd.set_option("max_columns", None)  # show all cols
pd.set_option('max_colwidth', None)  # show full width of showing cols
pd.set_option("expand_frame_repr", False)  #
# --

# Importing Dataset
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
per_new = per.copy()  # DF copy

# DF cleaning, preprocessing
per_new.rename(columns={'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'}, inplace=True)
per_new['University Rating'] = per_new['University Rating'].astype('category')
per_new = per_new.set_index('Serial No.')
per_new['GRE Groups'] = pd.cut(per_new['GRE Score'], 4, labels=[1, 2, 3, 4])  # Bining GRE into categories
per_new['TOEFL Groups'] = pd.cut(per_new['TOEFL Score'], 4, labels=[1, 2, 3, 4])  # Bining TOEFL into categories
per_new.head()
per_new.isna().any()

# Checking for Outliers with InterQuatile Range Detection for Int and Float

names = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA', 'Chance of Admit']

int_df = per_new.drop(['University Rating', 'Research', 'GRE Groups', 'TOEFL Groups'], axis=1)
n = 0
for e in range(1, 7):
    item = int_df.iloc[:, n].values
    q25 = np.percentile(item, 25)
    q50 = np.percentile(item, 50)
    q75 = np.percentile(item, 75)
    iqr = q75 - q25
    cutoff = iqr * 3  # k=3
    lower, upper = q25 - cutoff, q75 + cutoff
    print(names[n], end='')
    print(np.where(item > upper) or np.where(item < lower))  ## shows in which array the outlier appears
    n += 1

# per_new=per[(per.iloc[:,-1]>=0.01)&(per.iloc[:,-1]<=1)] #Chance of Admission
# per_new=per[(per.iloc[:,1]>=1)&(per.iloc[:,1]<=340)] #GRE
# per_new=per[(per.iloc[:,2]>=1)&(per.iloc[:,2]<=120)] #TOEFL
# per_new=per[(per.iloc[:,3]>=1)&(per.iloc[:,3]<=5)] #URating
# per_new=per[(per.iloc[:,4]>=1)&(per.iloc[:,4]<=5)] #SOP
# per_new=per[(per.iloc[:,5]>=1)&(per.iloc[:,5]<=5)] #LOR
# per_new=per[(per.iloc[:,6]>=1)&(per.iloc[:,6]<=10)] #CGPA
# per_new=per[(per.iloc[:,7]>=0)&(per.iloc[:,7]<=1)] #Research

# Checking for Outliers for categoricals

per_new['University Rating'].unique()
per_new['Research'].unique()

# Visualization
sns.heatmap(per.corr(), annot=True, cmap='summer')
plt.title("Correlation on Admission Features")
plt.tight_layout()
plt.show()
# ---------
sns.jointplot(x=per_new['CGPA'], y=per_new['Chance of Admit']);
plt.show()
sns.jointplot(x=per_new['CGPA'], y=per_new['Chance of Admit'], hue=per_new['Research']);
plt.show()
sns.jointplot(x=per_new['CGPA'], y=per_new['Chance of Admit'], hue=per_new['University Rating']);
plt.show()
sns.jointplot(x=per_new['GRE Score'], y=per_new['Chance of Admit'], hue=per_new['Research']);
plt.show()
sns.jointplot(x=per_new['GRE Score'], y=per_new['Chance of Admit'], hue=per_new['University Rating']);
plt.show()
sns.jointplot(x=per_new['TOEFL Score'], y=per_new['Chance of Admit'], hue=per_new['Research']);
plt.show()
sns.jointplot(x=per_new['TOEFL Score'], y=per_new['Chance of Admit'], hue=per_new['University Rating']);
plt.show()
sns.boxplot(x=per_new['Chance of Admit'], whis=np.inf);
plt.show()
# --As CGPA and GRE increases, chance of admission increases also.

# American dream selector
z = per_new.describe()
z.iloc[:, -1]
per['Admitted'] = per.iloc[:, -1].apply(lambda x: 0 if x < per.iloc[:, -1].mean() else 1)

# Modeling
x, y = per_new.drop(['Chance of Admit', 'GRE Groups', 'TOEFL Groups'], axis=1), per_new['Chance of Admit']
# The algorithm model that is going to learn from our data to make predictions
# splitting the data 80:20
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=4)

# Cross Validation to identify best model CONSTANT
models = ['Linear Reg', 'Decision Tree Reg', 'Random Forest Reg', 'Gradient Boosting Reg', 'Ada Boosting Reg',
          'Extra Tree Reg', 'K-Neighbors Reg', 'Support Vector Reg']

mod = [LinearRegression(), DecisionTreeRegressor(max_depth=5), RandomForestRegressor(n_estimators=50),
       GradientBoostingRegressor(), AdaBoostRegressor(), ExtraTreesRegressor(), KNeighborsRegressor(), SVR()]


def crossval(model_objs, x, y, mod_names_list):
    global df_cv
    score = []
    for i in model_objs:
        score.append(cross_val_score(i, x, y).mean())
    dic = dict(zip(mod_names_list, score))
    df_cv = pd.DataFrame(list(dic.items()), columns=['Model', 'Score'])


crossval(mod, x, y, models)
crosval_df = df_cv.copy()
crosval_df.sort_values(by='Score', inplace=True)
sns.barplot(y=crosval_df.Model, x=crosval_df.Score, data=df_cv, orient='h'), plt.tight_layout(), plt.show()


## america dream, its most likely get accepted into a good university
def myfunc(x, y):
    if x <= 0.75 and y > 2:  ## insert criteria here.
        return 1
    else:
        return 0


# df['American Dream'] = df.apply(lambda x: myfunc(x['Chance of Admit'],x['University Rating']), axis = 1)

# Cross Validation to identity best models CATEGORICAL

models2 = ['Logical Regression', 'Decision Tree Class', 'Random Forest Class', 'Gradient Boosting Class',
           'Ada Boosting Class',
           'Extra Tree Class', 'K-Neighbors Class', 'Support Vector Class']

mod2 = [LogisticRegression(), DecisionTreeClassifier(max_depth=5), RandomForestClassifier(n_estimators=50),
        GradientBoostingClassifier(), AdaBoostClassifier(), ExtraTreesClassifier(), KNeighborsClassifier(), SVC()]

# Fitting Random Forest
rf = RandomForestRegressor()
# Number of trees in random forest
n_estimators = list(np.arange(10, 100, 10))
# Criteriion for the random forest
criterion = ['mse', 'mae']  ## if you put this as a parameters , takes a long time to run and performance goes down
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = list(np.arange(4, 12))
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]
rf_param = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap}
# Using RandomizedSearch CV to look for the best parameter for our Random Forest
from sklearn.model_selection import RandomizedSearchCV

rf_improve = RandomizedSearchCV(estimator=rf, param_distributions=rf_param, n_iter=100, cv=5)
rf_improve.fit(X_train, y_train)
rf_improve.score(X_test, y_test)
rf_improve.best_params_
# After performing the randomized search we got the parameters it got and put it in our random forest method
rf = RandomForestRegressor(n_estimators=80, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=10,
                           bootstrap=True,random_state=10)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

y_pred = rf.predict(X_test)

estimator = rf.estimators_[1]

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
random_f = export_graphviz(estimator, out_file=None,
                           feature_names=X_train.columns,
                           rounded=True, proportion=False,
                           precision=2, filled=True)
graph = graph_from_dot_data(random_f)
graph.write_pdf("random_forest.pdf")
webbrowser.open_new(r'random_forest.pdf')

# Decision Tree's Tree
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
# decision_tree = export_graphviz(regressor, out_file=None, feature_names = X_train.columns,filled = True )
# # proffesor settings for the decision tree dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=data.iloc[:, 1:5].columns, out_file=None)
# graph = graph_from_dot_data(decision_tree)
# graph.write_pdf("decision_tree_gini.pdf")
# webbrowser.open_new(r'decision_tree_gini.pdf')
# rf.predict_proba(X_test)

from sklearn import metrics

print(f'Random Forest Regressor Score: {np.round(rf.score(X_test, y_test), 2) * 100}%')
print('RF Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('RF Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('RF Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))


##Putting our Randomizedsesearch into a datafram to check any trends
results = pd.DataFrame(rf_improve.cv_results_)
results[['params', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score')

## Feature Importance for our Random Forest
feature_names = x.columns
c = ['red', 'yellow', 'black', 'blue', 'orange', 'green', 'purple']
importance_Rf = pd.DataFrame(rf.feature_importances_, index=feature_names)
# importance_Rf.plot(kind='bar',color =c); plt.show()
plt.bar(x=feature_names, height=rf.feature_importances_, color=c)
plt.xticks(rotation=90)  ## this is the new graph
plt.tight_layout()
plt.show()

# Fitting Decision Tree to Training set
regressor = DecisionTreeRegressor(max_depth=5, random_state=10)  ##changing the depth because of our depth plot analysis
regressor.fit(X_train, y_train)
# Feature importance
feature_names = x.columns
feature_importance = pd.DataFrame(regressor.feature_importances_, index=feature_names)
print(feature_importance.sort_values)
## feature importance plot
features = list(feature_importance.index)
c = ['red', 'yellow', 'black', 'blue', 'orange', 'green', 'purple']
importance_df = pd.DataFrame(regressor.feature_importances_, index=feature_names)
plt.bar(x=feature_names, height=regressor.feature_importances_, color=c)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# feature_importance.plot(kind='bar'); plt.show()
# feature_importance.drop(['CGPA'],axis=0).plot(kind='bar'); plt.show()
# Prediction
pred = regressor.predict(X_test)
test_pred = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
# Disbribution of predicted vs actual change of admittion
sns.kdeplot(data=test_pred, x='Actual', label='Actual', color='olive')
sns.kdeplot(data=test_pred, x='Predicted', label='Predicted', color='teal')
plt.xlabel("Chance of Admit")
plt.legend()
plt.show()

# checking the different settings on the decision tree, first depth
train_accuracy = []
test_accuracy = []
##checking the different settings on the decision tree, first depth
for depth in range(1, 20):
    regressor2 = DecisionTreeRegressor(max_depth=depth, random_state=10)
    regressor2.fit(X_train, y_train)
    train_accuracy.append(regressor2.score(X_train, y_train))
    test_accuracy.append(regressor2.score(X_test, y_test))

frame = pd.DataFrame({'max_depth': range(1, 20), 'train_acc': train_accuracy, 'test_acc': test_accuracy})
frame.head(20)

# plotting the difference range in depth and visualizing which is one is better
plt.plot(range(1, 20), train_accuracy, marker='o', label='train_acc')  ##finding the appropiate depth
plt.plot(range(1, 20), test_accuracy, marker='o', label='test_acc')
plt.xlabel('Depth of tree')
plt.ylabel('Performance')
plt.legend()
plt.show()
# ---
plt.barh(frame['max_depth'], frame['train_acc'], color='red')
plt.barh(frame['max_depth'], -frame['test_acc'], color='green')
plt.ylabel('Depth of tree')
plt.title("TEST-------------- TRAIN")
plt.show()
# ---

# Metric
from sklearn import metrics

score = regressor.score(X_test, y_test)
print(f'DecisionTree Regressor Score: {np.round(score, 3) * 100}%')
print('DT Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred))
print('DT Mean Squared Error:', metrics.mean_squared_error(y_test, pred))
print('DT Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))



# Decision Tree's Tree
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
decision_tree = export_graphviz(regressor, out_file=None, feature_names=X_train.columns, filled=True)
# proffesor settings for the decision tree dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=data.iloc[:, 1:5].columns, out_file=None)
graph = graph_from_dot_data(decision_tree)
graph.write_pdf("decision_tree_gini.pdf")
webbrowser.open_new(r'decision_tree_gini.pdf')

# LinearRegression with SCALING
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

lr = LinearRegression()
scaler = StandardScaler()
scaler.fit(X_train.iloc[:, [0, 1, 3, 4, 5]])
scaled_data = scaler.transform(X_train.iloc[:, [0, 1, 3, 4, 5]])  # transforming only the numerical columns
df = pd.DataFrame(scaled_data,
                  columns=['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA'])  # dataframe with the scaling features
# getting_dummies = pd.get_dummies(X_train,columns=['University Rating','Research']) #gettig the dummies for University rating and researrcg
# getting_dummies = getting_dummies.iloc[:,5:12] #adding only the dummies to our dt
UR_Research = X_train.iloc[:, [2, 6]]

new_X_train = pd.concat([df.reset_index(drop=True), UR_Research.reset_index(drop=True)],
                        axis=1)  # reseting the index to concat two dataframe

scaler.fit(X_test.iloc[:, [0, 1, 3, 4, 5]])
scaled_data = scaler.transform(X_test.iloc[:, [0, 1, 3, 4, 5]])  # transforming only the numerical columns
df = pd.DataFrame(scaled_data,
                  columns=['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA'])  # dataframe with the scaling features
# getting_dummies = pd.get_dummies(X_test,columns=['University Rating','Research']) #gettig the dummies for University rating and researrcg
# getting_dummies = getting_dummies.iloc[:,5:12] #adding only the dummies to our dt
UR_Research2 = X_test.iloc[:, [2, 6]]
new_X_test = pd.concat([df.reset_index(drop=True), UR_Research2.reset_index(drop=True)], axis=1)

new_y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
new_y_test = scaler.fit_transform(y_test.values.reshape(-1, 1))

# new_X_train = scaler.fit_transform(X_train)
# new_X_test = scaler.fit_transform(X_test)
lr.fit(new_X_train, new_y_train)

pred1 = lr.predict(new_X_test)

print('Linear Regression Scaling Regression: ', lr.score(new_X_test, new_y_test))
print('LR Mean Absolute Error:', metrics.mean_absolute_error(pred1, new_y_test))
print('LR Mean Squared Error:', metrics.mean_squared_error(pred1, new_y_test))
print('LR Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(pred1, new_y_test)))


importances = pd.DataFrame(data={
    'Attribute': new_X_train.columns,
    'Importance': lr.coef_[0]
})  # this was copy from the internet to show the coefficients
importances = importances.sort_values(by='Importance', ascending=False)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()

## Linear Regression without SCALING
lr.fit(X_train, y_train)
lr.get_params()

# Linear Regression Coefficients
lr_coef = pd.DataFrame({'features': x.columns, 'coefficients': lr.coef_})
# Linear Regression Prediction
pred2 = lr.predict(X_test)
lr_pred = pd.DataFrame({'y_test': y_test, 'y_pred': pred2})
sns.kdeplot(data=lr_pred, x='y_test', label='Actual', color='olive')
sns.kdeplot(data=lr_pred, x='y_pred', label='Predicted', color='teal')
#sns.scatterplot(lr_pred['y_test'], lr_pred['y_pred'], color='blue')
plt.xlabel("Chance of Admit")
plt.legend()
plt.show()
score2 = lr.score(X_test, y_test)
print(f'Linear Regression Model Score: {np.round(score2, 3)*100}%')
print('LR Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred2))
print('LR Mean Squared Error:', metrics.mean_squared_error(y_test, pred2))
print('LR Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred2)))


## VOTING REGRESSOR

model_1 = LinearRegression()
model_2 = RandomForestRegressor(n_estimators=80, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                                max_depth=10, bootstrap=True)
model_3 = DecisionTreeRegressor(max_depth=5)
voting_essemble = VotingRegressor(estimators=[('lr', model_1), ('rf', model_2), ('dt', model_3)])
voting_essemble.fit(X_train, y_train)

print('Voting Essemble Score: ', voting_essemble.score(X_test, y_test))

# THIS DOESNT WORK!
# Residuals plot is used to analyze the variance of the error of the regressor.
# Residual Plot- Randomly scattered shows our linear model is good otherwise a non-linear model is preferred
# train=plt.scatter(lr.predict(y_train),(lr.predict(y_train)-y_train),color='green')
# test=plt.scatter(lr.predict(y_test),(lr.predict(y_test)-y_test),color='green')
# #plt.hlines(y=0,xmin=-10,xmax=10)
# plt.legend((train,test),('Training','Test'),loc='lower left')
# plt.title("Residual Plots")
# DONT delete this
# sns.scatterplot(y_test,pred,color='green')
# plt.xlabel("Actual Chance of Admit")
# plt.ylabel("Predicted Chance of Admit")
# plt.title("Scatter Plot for Predicted vs Actual Chance of Admit [Decision Tree]")
# plt.show()
