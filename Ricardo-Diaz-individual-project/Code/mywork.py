#*_________________________________________________________________________________*
### RICARDO CODE PYTHON PROJECT DATS6103

#Fitting Random Forest
rf = RandomForestRegressor()
# Number of trees in random forest
n_estimators = list(np.arange(10,100,10))
# Criteriion for the random forest
criterion = ['mse', 'mae'] ## if you put this as a parameters , takes a long time to run and performance goes down
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = list(np.arange(4,12))
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
rf_improve = RandomizedSearchCV(estimator = rf, param_distributions = rf_param,n_iter=100,cv = 5)
rf_improve.fit(X_train,y_train)
rf_improve.score(X_test,y_test)
rf_improve.best_params_
#After performing the randomized search we got the parameters it got and put it in our random forest method
rf = RandomForestRegressor(n_estimators= 80,min_samples_split = 2,min_samples_leaf = 1,max_features = 'sqrt',max_depth = 10,bootstrap = True)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)

y_pred = rf.predict(X_test)

estimator = rf.estimators_[1]

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
random_f = export_graphviz(estimator, out_file=None,
                feature_names = X_train.columns,
                rounded = True, proportion = False,
                precision = 2, filled = True)
graph = graph_from_dot_data(random_f)
graph.write_pdf("random_forest.pdf")
webbrowser.open_new(r'random_forest.pdf')

print(f'Random Forest Regressor Score: {np.round(rf.score(X_test,y_test),2)*100}%')
print('Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('Mean Squared Error:', mean_squared_error(y_test,y_pred))


# Fitting Decision Tree to Training set
regressor = DecisionTreeRegressor(max_depth=5,random_state=10) ##changing the depth because of our depth plot analysis
regressor.fit(X_train,y_train)
# Feature importance
feature_names = x.columns
feature_importance = pd.DataFrame(regressor.feature_importances_, index = feature_names)
print(feature_importance.sort_values)
## feature importance plot
features = list(feature_importance.index)
c = ['red', 'yellow', 'black', 'blue', 'orange','green','purple']
importance_df = pd.DataFrame(regressor.feature_importances_, index = feature_names)
plt.bar(x=feature_names,height=regressor.feature_importances_, color = c)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
#feature_importance.plot(kind='bar'); plt.show()
#feature_importance.drop(['CGPA'],axis=0).plot(kind='bar'); plt.show()
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
decision_tree = export_graphviz(regressor, out_file=None, feature_names = X_train.columns,filled = True )
# proffesor settings for the decision tree dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=data.iloc[:, 1:5].columns, out_file=None)
graph = graph_from_dot_data(decision_tree)
graph.write_pdf("decision_tree_gini.pdf")
webbrowser.open_new(r'decision_tree_gini.pdf')


# LinearRegression with SCALING
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

lr = LinearRegression()
scaler = StandardScaler()
scaler.fit(X_train.iloc[:,[0,1,3,4,5]])
scaled_data = scaler.transform(X_train.iloc[:,[0,1,3,4,5]]) #transforming only the numerical columns
df = pd.DataFrame(scaled_data, columns=['GRE Score' ,'TOEFL Score' ,'SOP','LOR','CGPA']) # dataframe with the scaling features
#getting_dummies = pd.get_dummies(X_train,columns=['University Rating','Research']) #gettig the dummies for University rating and researrcg
#getting_dummies = getting_dummies.iloc[:,5:12] #adding only the dummies to our dt
UR_Research = X_train.iloc[:,[2,6]]

new_X_train =  pd.concat([df.reset_index(drop=True),UR_Research.reset_index(drop=True)], axis=1) #reseting the index to concat two dataframe

scaler.fit(X_test.iloc[:,[0,1,3,4,5]])
scaled_data = scaler.transform(X_test.iloc[:,[0,1,3,4,5]]) #transforming only the numerical columns
df = pd.DataFrame(scaled_data, columns=['GRE Score' ,'TOEFL Score' ,'SOP','LOR','CGPA']) # dataframe with the scaling features
#getting_dummies = pd.get_dummies(X_test,columns=['University Rating','Research']) #gettig the dummies for University rating and researrcg
#getting_dummies = getting_dummies.iloc[:,5:12] #adding only the dummies to our dt
UR_Research2 = X_test.iloc[:,[2,6]]
new_X_test =  pd.concat([df.reset_index(drop=True),UR_Research2.reset_index(drop=True)], axis=1)

new_y_train = scaler.fit_transform(y_train.values.reshape(-1,1))
new_y_test = scaler.fit_transform(y_test.values.reshape(-1,1))

#new_X_train = scaler.fit_transform(X_train)
#new_X_test = scaler.fit_transform(X_test)

lr.fit(new_X_train,new_y_train)
print('Linear Regression Scaling Regression: ',lr.score(new_X_test,new_y_test))


importances = pd.DataFrame(data={
    'Attribute': new_X_train.columns,
    'Importance': lr.coef_[0]
}) #this was copy from the internet to show the coefficients
importances = importances.sort_values(by='Importance', ascending=False)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()



class RandomForest(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = 'Random Forest Regression'
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.setWindowIcon(QIcon('pty.png'))

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('RF Regression Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()
        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")

        self.lblN_estimators = QLabel('N Estimators :')
        self.lblN_estimators.adjustSize()
        self.txtN_estimators = QLineEdit(self)
        self.txtN_estimators.setText("80")

        self.lblMin_sample_split = QLabel('Min sample split :')
        self.lblMin_sample_split.adjustSize()
        self.txtMin_sample_split = QLineEdit(self)
        self.txtMin_sample_split.setText("2")

        # self.lblRandomState = QLabel('Max Features :')
        # self.lblRandomState.adjustSize()
        # self.txtRandomState = QLineEdit(self)
        # self.txtRandomState.setText("sqrt")

        self.lblMin_samples_leaf = QLabel('Min samples leaf :')
        self.lblMin_samples_leaf.adjustSize()
        self.txtMin_samples_leaf = QLineEdit(self)
        self.txtMin_samples_leaf.setText("1")

        self.lblMax_depth = QLabel('Max depth :')
        self.lblMax_depth.adjustSize()
        self.txtMax_depth = QLineEdit(self)
        self.txtMax_depth.setText("10")

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.setShortcut("Ctrl+E")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)
        self.groupBox1Layout.addWidget(self.feature6, 3, 0)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.lblN_estimators, 4, 2)
        self.groupBox1Layout.addWidget(self.txtN_estimators, 4, 3)
        self.groupBox1Layout.addWidget(self.lblMin_sample_split, 5, 0)
        self.groupBox1Layout.addWidget(self.txtMin_sample_split, 5, 1)
        self.groupBox1Layout.addWidget(self.lblMin_samples_leaf, 5, 2)
        self.groupBox1Layout.addWidget(self.txtMin_samples_leaf, 5, 3)
        self.groupBox1Layout.addWidget(self.lblMax_depth, 6, 0)
        self.groupBox1Layout.addWidget(self.txtMax_depth, 6, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 6, 2)

        self.groupBox2 = QGroupBox('Actual vs Predicted')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Dataframe:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy Score:')
        self.txtAccuracy = QLineEdit()
        self.lblR2 = QLabel('R Square:')
        self.txtR2 = QLineEdit()
        self.lblR3 = QLabel('MSE:')
        self.txtR3 = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)
        self.groupBox2Layout.addWidget(self.lblR2)
        self.groupBox2Layout.addWidget(self.txtR2)
        self.groupBox2Layout.addWidget(self.lblR3)
        self.groupBox2Layout.addWidget(self.txtR3)

        # Graphic: Importance of Features
        #::-------------------------------------------
        self.fig1 = Figure()
        self.ax3 = self.fig1.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas = FigureCanvas(self.fig1)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG3 = QGroupBox('Feature Importance')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas)

        self.layout.addWidget(self.groupBox1, 0, 0)
        # self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2, 0, 2)
        # self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3, 1, 0)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        # We process the parameters`
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[3]]], axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[5]]], axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[6]]], axis=1)

        vtest_per = float(self.txtPercentTest.text())
        self.txtR3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # We assign the values to X and y to run the algorithm

        X_dt = self.list_corr_features
        y_dt = per_new["Chance of Admit"]

        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=2)

        n_estimators1 = int(self.txtN_estimators.text())
        min_samples_split2 = int(self.txtMin_sample_split.text())
        min_samples_leaf3 = int(self.txtMin_samples_leaf.text())
        # max_features4 = str(self.txtmax_features.text())
        max_depth5 = int(self.txtMax_depth.text())

        self.rf = RandomForestRegressor(n_estimators=n_estimators1, min_samples_split=min_samples_split2,
                                        min_samples_leaf=min_samples_leaf3, max_features='sqrt',
                                        max_depth=max_depth5, bootstrap=True)  ## set all in initUI

        # performing training and prediction
        self.rf.fit(X_train, y_train)
        y_pred = self.rf.predict(X_test)

        # prediction
        self.rf_pred_df = pd.DataFrame({'y_test': y_test, 'y_pred': np.round(y_pred, 2)})

        # accuracy

        self.rf_accuracy_score = self.rf.score(X_test, y_test)
        self.txtAccuracy.setText(str(np.round(self.rf_accuracy_score * 100, 3)) + '%')

        # R_squared
        self.r2 = r2_score(y_test, y_pred)
        self.txtR2.setText(str(np.round(self.r2 * 100)) + '%')
        # MSE
        self.mse = mean_squared_error(y_test, y_pred)
        self.txtR3.setText(str(np.round(self.mse * 100, 4)) + '%')

        # Feature importance
        feature_names = ['GRE', 'TOEFL', 'URating', 'SOP', 'LOR', 'CGPA', 'Research']
        feature_imp = self.rf.feature_importances_

        feature_importance = pd.DataFrame(self.rf.feature_importances_, index=feature_names, columns=['score'])
        self.txtResults.appendPlainText(self.rf_pred_df.to_csv(sep="|", index=False))
        self.txtResults.updateGeometry()
        c = ['red', 'yellow', 'black', 'blue', 'orange', 'green', 'purple']
        # ---test , try to do a barplot
        self.ax3.clear()
        # self.ax3.bar(x=feature_names, height=feature_imp, color=c)
        sns.barplot(x=feature_names, y=feature_imp, ax=self.ax3, palette='Blues_d')
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()


class DecisionTree(QMainWindow):
    #::----------------------
    # Implementation of Decision Tree Algorithm using the admission dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parameters
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree, self).__init__()

        self.Title = "Decision Tree Regression"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.setWindowIcon(QIcon('pty.png'))

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Linear Regression Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()
        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")

        self.lblRandomState = QLabel('Random State :')
        self.lblRandomState.adjustSize()
        self.txtRandomState = QLineEdit(self)
        self.txtRandomState.setText("2")

        self.btnExecute = QPushButton("Execute DTR")
        self.btnExecute.setShortcut("Ctrl+E")
        self.btnExecute.clicked.connect(self.update)

        self.btnDTFigure = QPushButton("View Tree")
        self.btnDTFigure.setShortcut('Ctrl+T')
        self.btnDTFigure.clicked.connect(self.view_tree)
        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)
        self.groupBox1Layout.addWidget(self.feature6, 3, 0)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.lblRandomState, 4, 2)
        self.groupBox1Layout.addWidget(self.txtRandomState, 4, 3)
        self.groupBox1Layout.addWidget(self.btnExecute, 6, 0)
        self.groupBox1Layout.addWidget(self.btnDTFigure, 6, 1)

        self.groupBox2 = QGroupBox('Actual vs Predicted')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Dataframe:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy Score:')
        self.txtAccuracy = QLineEdit()
        self.lblR2 = QLabel('R Square:')
        self.txtR2 = QLineEdit()
        self.lblR3 = QLabel('MSE:')
        self.txtR3 = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)
        self.groupBox2Layout.addWidget(self.lblR2)
        self.groupBox2Layout.addWidget(self.txtR2)
        self.groupBox2Layout.addWidget(self.lblR3)
        self.groupBox2Layout.addWidget(self.txtR3)

        # Graphic: Importance of Features
        #::-------------------------------------------

        self.fig1 = Figure()
        self.ax3 = self.fig1.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas = FigureCanvas(self.fig1)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG3 = QGroupBox('Feature Importance')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas)

        # -----------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 0)
        # self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2, 0, 2)
        # self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3, 1, 0)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Decision Tree Algorithm
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Decision Tree algorithm
          then the results are presented in graphics and reports in the canvas
        :return: None
        '''

        # We process the parameters`
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[3]]], axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[5]]], axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[6]]], axis=1)

        vtest_per = float(self.txtPercentTest.text())
        self.txtR3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # We assign the values to X and y to run the algorithm

        X_dt = self.list_corr_features
        y_dt = per_new["Chance of Admit"]

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=2)
        # perform training with mse.
        randomState = int(self.txtRandomState.text())
        self.dtr = DecisionTreeRegressor(max_depth=5, random_state=randomState)

        # Performing training
        self.dtr.fit(X_train, y_train)

        # prediction on test
        y_pred = self.dtr.predict(X_test)

        self.dtr_pred_df = pd.DataFrame({'y_test': y_test, 'y_pred': np.round(y_pred, 2)})

        # accuracy score
        self.ff_accuracy_score = self.dtr.score(X_test, y_test)
        self.txtAccuracy.setText(str(np.round(self.ff_accuracy_score * 100, 3)) + '%')

        # R squared
        self.r2 = r2_score(y_test, y_pred)
        self.txtR2.setText(str(np.round(self.r2 * 100)) + '%')

        # MSE
        self.mse = mean_squared_error(y_test, y_pred)
        self.txtR3.setText(str(np.round(self.mse * 100, 4)) + '%')

        # Feature importance
        feature_names = self.list_corr_features.columns
        feature_importance = pd.DataFrame(self.dtr.feature_importances_, index=feature_names, columns=['score'])
        self.txtResults.appendPlainText(self.dtr_pred_df.to_csv(sep="|", index=False))
        self.txtResults.updateGeometry()

        # ---test
        self.ax3.clear()
        # self.ax3.barh(feature_names,list(feature_importance['score']))
        self.ax3.plot(feature_names, list(feature_importance['score']), 'o-', color='g')
        self.ax3.tick_params(axis='x', labelsize=7)
        self.ax3.set_aspect('auto')
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

    def view_tree(self):
        dot_data = export_graphviz(self.dtr, out_file=None, feature_names=self.list_corr_features.columns, filled=True,
                                   max_depth=4)
        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_regressor.pdf")
        webbrowser.open_new(r'decision_tree_regressor.pdf')
