import sys

from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit, QTableView)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QAbstractTableModel, Qt
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
#from scipy import interp
from scipy import interpolate
from itertools import cycle
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
#---
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,GradientBoostingRegressor,
                              AdaBoostRegressor,ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR,SVC
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import export_graphviz
import os
import warnings
warnings.filterwarnings("ignore")
from pydotplus import graph_from_dot_data


class RandomForest(QMainWindow):

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.title = 'Random Forest Regression'
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

        #self.lblRandomState = QLabel('Max Features :')
        #self.lblRandomState.adjustSize()
        #self.txtRandomState = QLineEdit(self)
        #self.txtRandomState.setText("sqrt")

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


        self.layout.addWidget(self.groupBox1,0,0)
        #self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)
        # self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3,1,0)

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

        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per,random_state=2)

        n_estimators1 = int(self.txtN_estimators.text())
        min_samples_split2 = int(self.txtMin_sample_split.text())
        min_samples_leaf3 = int(self.txtMin_samples_leaf.text())
        #max_features4 = str(self.txtmax_features.text())
        max_depth5 = int(self.txtMax_depth.text())

        self.rf = RandomForestRegressor(n_estimators=n_estimators1, min_samples_split=min_samples_split2, min_samples_leaf=min_samples_leaf3, max_features='sqrt',
                                   max_depth=max_depth5, bootstrap=True) ## set all in initUI

        #performing training and prediction
        self.rf.fit(X_train, y_train)
        y_pred = self.rf.predict(X_test)

        #prediction
        self.rf_pred_df = pd.DataFrame({'y_test': y_test,'y_pred': np.round(y_pred,2)})

        #accuracy

        self.rf_accuracy_score = self.rf.score(X_test,y_test)
        self.txtAccuracy.setText(str(np.round(self.rf_accuracy_score*100,3))+'%')

        #R_squared
        self.r2 = r2_score(y_test,y_pred)
        self.txtR2.setText(str(np.round(self.r2*100))+'%')
        #MSE
        self.mse = mean_squared_error(y_test,y_pred)
        self.txtR3.setText(str(np.round(self.mse*100, 4))+'%')

        # Feature importance
        feature_names = self.list_corr_features.columns
        feature_importance = pd.DataFrame(self.rf.feature_importances_, index=feature_names, columns=['score'])
        self.txtResults.appendPlainText(self.rf_pred_df.to_csv(sep="|", index=False))
        self.txtResults.updateGeometry()

        # ---test , try to do a barplot
        self.ax3.clear()
        # self.ax3.barh(feature_names,list(feature_importance['score']))
        self.ax3.plot(feature_names, list(feature_importance['score']), 'o-', color='g')
        self.ax3.tick_params(axis='x', labelsize=7)
        self.ax3.set_aspect('auto')
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()
        ## Feature Importance for our Random Forest
        # feature_names = x.columns ##get the columns from the x
        # c = ['red', 'yellow', 'black', 'blue', 'orange', 'green', 'purple']
        # importance_Rf = pd.DataFrame(rf.feature_importances_, index=feature_names)
        # # importance_Rf.plot(kind='bar',color =c); plt.show()
        # plt.bar(x=feature_names, height=rf.feature_importances_, color=c)
        # plt.xticks(rotation=90)  ## this is the new graph
        # plt.tight_layout()
        # plt.show()


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

        self.Title ="Decision Tree Regression"
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
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
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

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0)
        self.groupBox1Layout.addWidget(self.lblPercentTest,4,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,4,1)
        self.groupBox1Layout.addWidget(self.lblRandomState,4,2)
        self.groupBox1Layout.addWidget(self.txtRandomState, 4, 3)
        self.groupBox1Layout.addWidget(self.btnExecute,6,0)
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

        #-----------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        #self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)
        # self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3,1,0)

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
            if len(self.list_corr_features)==0:
                self.list_corr_features = per_new[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[6]]],axis=1)


        vtest_per =float(self.txtPercentTest.text())
        self.txtR3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per/100


        # We assign the values to X and y to run the algorithm

        X_dt =  self.list_corr_features
        y_dt = per_new["Chance of Admit"]

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per,random_state=2)
        # perform training with mse.
        randomState= int(self.txtRandomState.text())
        self.dtr = DecisionTreeRegressor(max_depth=5, random_state=randomState)

        # Performing training
        self.dtr.fit(X_train, y_train)

        # prediction on test
        y_pred = self.dtr.predict(X_test)

        self.dtr_pred_df = pd.DataFrame({'y_test':y_test,'y_pred':np.round(y_pred,2)})


        # accuracy score
        self.ff_accuracy_score = self.dtr.score(X_test, y_test)
        self.txtAccuracy.setText(str(np.round(self.ff_accuracy_score*100,3))+'%')

        #R squared
        self.r2=r2_score(y_test,y_pred)
        self.txtR2.setText(str(np.round(self.r2*100))+'%')

        #MSE
        self.mse= mean_squared_error(y_test,y_pred)
        self.txtR3.setText(str(np.round(self.mse*100, 4))+'%')

        # Feature importance
        feature_names = self.list_corr_features.columns
        feature_importance = pd.DataFrame(self.dtr.feature_importances_, index=feature_names, columns=['score'])
        self.txtResults.appendPlainText(self.dtr_pred_df.to_csv(sep="|", index=False))
        self.txtResults.updateGeometry()

        #---test
        self.ax3.clear()
        #self.ax3.barh(feature_names,list(feature_importance['score']))
        self.ax3.plot(feature_names,list(feature_importance['score']),'o-',color='g')
        self.ax3.tick_params(axis='x',labelsize=7)
        self.ax3.set_aspect('auto')
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

    def view_tree(self):
        dot_data = export_graphviz(self.dtr, out_file=None,feature_names=self.list_corr_features.columns,filled=True,max_depth=4)
        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_regressor.pdf")
        webbrowser.open_new(r'decision_tree_regressor.pdf')


class Regression(QMainWindow):
    #::----------------------
    # Implementation of Linear Regression Algorithm using the admission dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Regression, self).__init__()

        self.Title ="Linear Regression"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.setWindowIcon(QIcon('pty.png'))

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Linear Regression Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
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



        self.btnExecute = QPushButton("Execute LR")
        self.btnExecute.setShortcut("Ctrl+E")
        self.btnExecute.clicked.connect(self.update)


        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0)
        self.groupBox1Layout.addWidget(self.lblPercentTest,4,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,4,1)
        self.groupBox1Layout.addWidget(self.btnExecute,5,0)

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

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)
        self.groupBox2Layout.addWidget(self.lblR2)
        self.groupBox2Layout.addWidget(self.txtR2)

        # Graphic: PREDICTION VS ACTUAL

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes1 = [self.ax1]
        self.canvas = FigureCanvas(self.fig1)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG3 = QGroupBox('Prediction v Actual Distribution')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas)

        #-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBox2,0,2)
        self.layout.addWidget(self.groupBoxG3,1,0)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()


    def update(self):

        # We process the parameters`
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = per_new[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = per_new[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, per_new[features_list[6]]],axis=1)


        vtest_per =float(self.txtPercentTest.text())

        #self.ax1.clear()
        # self.ax2.clear()
        # self.ax3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per/100


        # We assign the values to X and y to run the algorithm

        X_dt =  self.list_corr_features
        y_dt = per_new["Chance of Admit"]

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per,random_state=2)
        # perform training with mse.
        # Decision tree with gini
        self.lr = LinearRegression()

        # Performing training
        self.lr.fit(X_train, y_train)

        # prediction on test
        y_pred = self.lr.predict(X_test)

        self.lr_pred = pd.DataFrame({'y_test':y_test,'y_pred':np.round(y_pred,2)})
        self.txtResults.appendPlainText(self.lr_pred.to_csv(sep="|",index=False))
        self.txtResults.updateGeometry()

        # accuracy score

        self.ff_accuracy_score = self.lr.score(X_test, y_test)
        self.txtAccuracy.setText(str(np.round(self.ff_accuracy_score*100,3))+'%')

        #R squared
        self.r2=r2_score(y_test,y_pred)
        self.txtR2.setText(str(np.round(self.r2*100))+'%')

        #---plot
        self.ax1.clear()
        xx=np.arange(len(y_pred))
        self.ax1.plot(xx,y_pred,color='red',lw=2,label="predicted",alpha=0.5)
        self.ax1.plot(xx,y_test,color='green',lw=2,label="actual",alpha=0.5)
        self.ax1.set_ylabel("Chance of Admission")
        self.ax1.legend(loc="lower right")
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()


class CorrelationPlot(QMainWindow):
    #;:-----------------------------------------------------------------------
    # This class creates a canvas to draw a correlation plot
    # It presents all the features plus the happiness score
    # the methods for this class are:
    #   _init_
    #   initUi
    #   update
    #::-----------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        #::--------------------------------------------------------
        super(CorrelationPlot, self).__init__()

        self.Title = 'Correlation Plot'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  Creates the canvas and elements of the canvas
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Correlation Plot Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)


        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature7 = QCheckBox(features_list[7], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        self.feature7.setChecked(True)

        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,0,2)
        self.groupBox1Layout.addWidget(self.feature3,0,3)
        self.groupBox1Layout.addWidget(self.feature4,1,0)
        self.groupBox1Layout.addWidget(self.feature5,1,1)
        self.groupBox1Layout.addWidget(self.feature6,1,2)
        self.groupBox1Layout.addWidget(self.feature7,1,3)
        self.groupBox1Layout.addWidget(self.btnExecute,2,0)


        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()


        self.groupBox2 = QGroupBox('Correlation Plot')
        self.groupBox2Layout= QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox2Layout.addWidget(self.canvas)


        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)

        self.setCentralWidget(self.main_widget)
        self.resize(900, 700)
        self.show()
        self.update()

    def update(self):

        #::------------------------------------------------------------
        # Populates the elements in the canvas using the values
        # chosen as parameters for the correlation plot
        #::------------------------------------------------------------
        self.ax1.clear()

        X_1 = ff_happiness["Happiness.Score"]

        list_corr_features = pd.DataFrame(ff_happiness["Happiness.Score"])
        if self.feature0.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_happiness[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_happiness[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_happiness[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_happiness[features_list[3]]],axis=1)
        if self.feature4.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_happiness[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_happiness[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_happiness[features_list[6]]],axis=1)

        if self.feature7.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_happiness[features_list[7]]],axis=1)


        vsticks = ["dummy"]
        vsticks1 = list(list_corr_features.columns)
        vsticks1 = vsticks + vsticks1
        res_corr = list_corr_features.corr()
        self.ax1.matshow(res_corr, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(vsticks1)
        self.ax1.set_xticklabels(vsticks1,rotation = 90)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
