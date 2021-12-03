""" Introduction to Data Mining Project
    Group-4
    Names- Ricardo, Junran, Jeremiah, Osemekhian
"""

######################################################################
# ---
import sys

from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit,
                             QTableView)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QAbstractTableModel, Qt
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
# from scipy import interp
from scipy import interpolate
from itertools import cycle
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# ---
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import export_graphviz
import os
import warnings

warnings.filterwarnings("ignore")
from pydotplus import graph_from_dot_data
import webbrowser

# ---
font_size_window = 'font-size:15px'
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
os.environ['KAGGLE_USERNAME'] = 'koyanjo'
os.environ['KAGGLE_KEY'] = '33bfba07e0815efc297a1a4488dbe6a3'

from kaggle.api.kaggle_api_extended import KaggleApi

dataset = 'mohansacharya/graduate-admissions'
path = 'datasets/graduate-admissions'
api = KaggleApi()
api.authenticate()
api.dataset_download_files(dataset, path)
api.dataset_download_file(dataset, 'Admission_Predict.csv', path)


# ---

class CorrelationPlot(QMainWindow):
    # ;:-----------------------------------------------------------------------
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

        self.Title = 'Heatmap Plot'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  Creates the canvas and elements of the canvas
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.setWindowIcon(QIcon('pty.png'))

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Heatmap Plot for Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)
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

        self.btnExecute = QPushButton("Creat Heatmap")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 0, 2)
        self.groupBox1Layout.addWidget(self.feature3, 0, 3)
        self.groupBox1Layout.addWidget(self.feature4, 1, 0)
        self.groupBox1Layout.addWidget(self.feature5, 1, 1)
        self.groupBox1Layout.addWidget(self.feature6, 1, 2)
        self.groupBox1Layout.addWidget(self.feature7, 1, 3)
        self.groupBox1Layout.addWidget(self.btnExecute, 2, 0)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBox2 = QGroupBox('Heatmap Plot')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox2Layout.addWidget(self.canvas)

        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 900)
        self.show()
        self.update()

    def update(self):

        #::------------------------------------------------------------
        # Populates the elements in the canvas using the values
        # chosen as parameters for the Heatmap plot
        #::------------------------------------------------------------
        self.ax1.clear()

        X_1 = per_new["Chance of Admit"]

        list_corr_features = pd.DataFrame(per_new["Chance of Admit"])
        if self.feature0.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[3]]], axis=1)
        if self.feature4.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[5]]], axis=1)

        if self.feature6.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[6]]], axis=1)

        if self.feature7.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[7]]], axis=1)

        vsticks = ["dummy"]
        vsticks1 = list(list_corr_features.columns)
        vsticks1 = vsticks + vsticks1
        res_corr = list_corr_features.corr()
        sns.heatmap(res_corr,annot=True,cbar=False, ax=self.ax1)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


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


class cross(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(cross, self).__init__()

        self.Title = "Cross Validation"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.setWindowIcon(QIcon('pty.png'))

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Model Score')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.lbl1 = QLabel('Linear Regression :')
        self.txt1 = QLineEdit(self)
        self.txt1.setText(str(np.round(cross_val_score(LinearRegression(), x, y).mean() * 100, 2)) + '%')
        self.lbl2 = QLabel('Decision Tree Reg :')
        self.txt2 = QLineEdit(self)
        self.txt2.setText(
            str(np.round(cross_val_score(DecisionTreeRegressor(random_state=2), x, y).mean() * 100, 2)) + '%')
        self.lbl3 = QLabel('Random Forest Reg :')
        self.txt3 = QLineEdit(self)
        self.txt3.setText(
            str(np.round(cross_val_score(RandomForestRegressor(n_estimators=50, random_state=2), x, y).mean() * 100,
                         2)) + '%')
        self.lbl4 = QLabel('Gradient Boosting Reg :')
        self.txt4 = QLineEdit(self)
        self.txt4.setText(str(np.round(cross_val_score(GradientBoostingRegressor(), x, y).mean() * 100, 2)) + '%')
        self.lbl5 = QLabel('Ada Boosting Reg :')
        self.txt5 = QLineEdit(self)
        self.txt5.setText(str(np.round(cross_val_score(AdaBoostRegressor(), x, y).mean() * 100, 2)) + '%')
        self.lbl6 = QLabel('Extra Tree Reg :')
        self.txt6 = QLineEdit(self)
        self.txt6.setText(str(np.round(cross_val_score(ExtraTreesRegressor(), x, y).mean() * 100, 2)) + '%')
        self.lbl7 = QLabel('KNeighbors Reg :')
        self.txt7 = QLineEdit(self)
        self.txt7.setText(str(np.round(cross_val_score(KNeighborsRegressor(), x, y).mean() * 100, 2)) + '%')
        self.lbl8 = QLabel('Support Vector Reg :')
        self.txt8 = QLineEdit(self)
        self.txt8.setText(str(np.round(cross_val_score(SVR(), x, y).mean() * 100, 2)) + '%')

        # We create a checkbox for each model
        self.groupBox1Layout.addWidget(self.lbl1, 0, 0)
        self.groupBox1Layout.addWidget(self.txt1, 0, 1)
        self.groupBox1Layout.addWidget(self.lbl2, 1, 0)
        self.groupBox1Layout.addWidget(self.txt2, 1, 1)
        self.groupBox1Layout.addWidget(self.lbl3, 2, 0)
        self.groupBox1Layout.addWidget(self.txt3, 2, 1)
        self.groupBox1Layout.addWidget(self.lbl4, 3, 0)
        self.groupBox1Layout.addWidget(self.txt4, 3, 1)
        self.groupBox1Layout.addWidget(self.lbl5, 4, 0)
        self.groupBox1Layout.addWidget(self.txt5, 4, 1)
        self.groupBox1Layout.addWidget(self.lbl6, 5, 0)
        self.groupBox1Layout.addWidget(self.txt6, 5, 1)
        self.groupBox1Layout.addWidget(self.lbl7, 6, 0)
        self.groupBox1Layout.addWidget(self.txt7, 6, 1)
        self.groupBox1Layout.addWidget(self.lbl8, 7, 0)
        self.groupBox1Layout.addWidget(self.txt8, 7, 1)

        # testy
        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes1 = [self.ax1]
        self.canvas = FigureCanvas(self.fig1)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        # testy

        self.groupBox2 = QGroupBox('Model Scores')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2Layout.addWidget(self.canvas)

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBox2, 0, 1)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

        # ---test
        self.ax1.clear()
        # xx = np.arange(len(y_pred))
        #self.ax1.plot(cval.index, cval['score'], marker='o', label="Model Score", color='green')
        c = ['red', 'yellow', 'black', 'blue', 'orange', 'green', 'purple']
        # self.ax1.bar(x=cval.index, height=cval['score'],color=c)
        sns.barplot(y=cval.index, x=cval['score'],ax=self.ax1,orient='h',palette='Blues_d')
        self.ax1.tick_params(axis='x', labelsize=7)
        # self.ax1.plot(xx, y_test, color='green', lw=2, label="actual", alpha=0.5)
        self.ax1.set_ylabel("Model's Mean Score")
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()


# test
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

        self.Title = "Linear Regression"
        self.initUi()

    def initUi(self):

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

        self.btnExecute = QPushButton("Execute LR")
        self.btnExecute.setShortcut("Ctrl+E")
        self.btnExecute.clicked.connect(self.update)

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
        self.groupBox1Layout.addWidget(self.btnExecute, 5, 0)

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

        # -------------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBox2, 0, 2)
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

        # self.ax1.clear()
        # self.ax2.clear()
        # self.ax3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # We assign the values to X and y to run the algorithm

        X_dt = self.list_corr_features
        y_dt = per_new["Chance of Admit"]

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=2)
        # perform training with mse.
        # Decision tree with gini
        self.lr = LinearRegression()

        # Performing training
        self.lr.fit(X_train, y_train)

        # prediction on test
        y_pred = self.lr.predict(X_test)

        self.lr_pred = pd.DataFrame({'y_test': y_test, 'y_pred': np.round(y_pred, 2)})
        self.txtResults.appendPlainText(self.lr_pred.to_csv(sep="|", index=False))
        self.txtResults.updateGeometry()

        # accuracy score

        self.ff_accuracy_score = self.lr.score(X_test, y_test)
        self.txtAccuracy.setText(str(np.round(self.ff_accuracy_score * 100, 3)) + '%')

        # R squared
        self.r2 = r2_score(y_test, y_pred)
        self.txtR2.setText(str(np.round(self.r2 * 100)) + '%')

        # ---plot
        self.ax1.clear()
        sns.kdeplot(x=y_test,ax=self.ax1,shade=True,label="actual")
        sns.kdeplot(x=y_pred,ax=self.ax1,shade=True,label="predicted")
        self.ax1.set_ylabel("Chance of Admission")
        self.ax1.legend(loc="upper left")
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()


class AdmitGraphs(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a plot to show the relation
    # from each feature in the dataset with the happiness score
    # methods
    #    _init_
    #   update
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout to draw a plot
        super(AdmitGraphs, self).__init__()

        self.Title = "Features vrs Admit Chance"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.setWindowIcon(QIcon('pty.png'))

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA',
                                 'Research'])

        self.dropdown1.currentIndexChanged.connect(self.update)
        self.label = QLabel("A plot:")

        self.checkbox1 = QCheckBox('Show Regression Line', self)
        self.checkbox1.stateChanged.connect(self.update)

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select Index for subplots"))
        self.layout.addWidget(self.dropdown1)
        self.layout.addWidget(self.checkbox1)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.show()
        self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        # containing the elements of the graph
        self.ax1.clear()
        cat1 = self.dropdown1.currentText()

        X_1 = per_new["Chance of Admit"]
        y_1 = per_new[cat1]

        sns.scatterplot(x=X_1, y=y_1,data=per_new,palette='deep',legend='full',ax=self.ax1)

        if self.checkbox1.isChecked():
            b, m = polyfit(X_1, y_1, 1)

            self.ax1.plot(X_1, b + m * X_1, '-', color="green")

        vtitle = "Chance of Admission vrs " + cat1
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel("Chance of Admission")
        self.ax1.set_ylabel(cat1)
        self.ax1.grid(True)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


##test
class box(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a Boxplot
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout to draw a plot
        super(box, self).__init__()

        self.Title = "BOXPLOT"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.setWindowIcon(QIcon('pty.png'))

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA'])

        self.dropdown1.currentIndexChanged.connect(self.update)
        self.label = QLabel("A plot:")

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(self.dropdown1)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.show()
        self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        # containing the elements of the graph
        self.ax1.clear()
        cat = self.dropdown1.currentText()
        sns.boxplot(per_new[cat],ax=self.ax1)
        vtitle = "Boxplot of "+cat
        self.ax1.set_title(vtitle)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

##test
class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw a histogram graph
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)


class CanvasWindow(QMainWindow):
    #::----------------------------------
    # Creates a canvaas containing the plot for the initial analysis
    def __init__(self, parent=None):
        super(CanvasWindow, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Distribution'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)


class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'Chance of Admission'
        self.width = 500
        self.height = 300
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowIcon(QIcon('raccoon.png'))

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightgreen')

        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('E.D.A Analysis')
        MLModelMenu = mainMenu.addMenu('ML Models')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # EDA analysis

        EDA1Button = QAction(QIcon('pty.png'), 'Distribution', self)
        EDA1Button.setStatusTip('Distribution of Chance of Admission')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        EDA2Button = QAction(QIcon('pty.png'), 'Scatter Plots', self)
        EDA2Button.setStatusTip('Twain features relationship')
        EDA2Button.triggered.connect(self.EDA2)
        EDAMenu.addAction(EDA2Button)

        EDA4Button = QAction(QIcon('pty.png'), 'Heatmap Plot', self)
        EDA4Button.setStatusTip('Features Correlation Plot')
        EDA4Button.triggered.connect(self.EDA4)
        EDAMenu.addAction(EDA4Button)

        EDA5Button = QAction(QIcon('pty.png'), 'Box Plot', self)
        EDA5Button.setStatusTip('Box Plot')
        EDA5Button.triggered.connect(self.EDA5)
        EDAMenu.addAction(EDA5Button)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are three models
        # Linear Regression
        #       Decision Tree
        #       Random Forest
        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------

        MLModel1Button = QAction(QIcon('pty.png'), 'Linear Regression', self)
        MLModel1Button.setStatusTip('Linear Regression')
        MLModelMenu.addAction(MLModel1Button)
        MLModel1Button.triggered.connect(self.MLLR)

        #::------------------------------------------------------
        # Decision Tree Regression
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon('pty.png'), 'Decision Tree Regression', self)
        MLModel2Button.setStatusTip('Decision Tree Regression')
        MLModelMenu.addAction(MLModel2Button)
        MLModel2Button.triggered.connect(self.MLDT)
        #::------------------------------------------------------
        # Cross Validation
        #::------------------------------------------------------
        MLModel3Button = QAction(QIcon('pty.png'), 'Cross Validation', self)
        MLModel3Button.setStatusTip('Cross Validation')
        MLModelMenu.addAction(MLModel3Button)
        MLModel3Button.triggered.connect(self.MLS)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------

        MLModel4Button = QAction(QIcon('pty.png'), 'Random Forest Regression', self)
        MLModel4Button.setStatusTip('Random Forest Regression ')
        MLModelMenu.addAction(MLModel4Button)
        MLModel4Button.triggered.connect(self.MLRF)

        #
        self.dialogs = list()

    def EDA1(self):
        #::------------------------------------------------------
        # Creates the histogram
        # The X variable contains the admit.score(chance)
        # x was populated in the method data_admit()
        # at the start of the application
        #::------------------------------------------------------
        dialog = CanvasWindow(self)
        dialog.m.plot()
        sns.histplot(y,kde='gau',ax=dialog.m.ax,bins=20)
        dialog.m.ax.set_title('Distribution of Chance of Admission')
        dialog.m.ax.set_xlabel("Chance of Admission")
        dialog.m.draw()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        #::---------------------------------------------------------
        dialog = AdmitGraphs()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA4(self):
        #::----------------------------------------------------------
        # This function creates an instance of the CorrelationPlot class
        #::----------------------------------------------------------
        dialog = CorrelationPlot()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA5(self):
        #::---------------------------------------------------------
        dialog = box()
        self.dialogs.append(dialog)
        dialog.show()

    def MLDT(self):
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

    def MLLR(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        #::-----------------------------------------------------------
        dialog = Regression()
        self.dialogs.append(dialog)
        dialog.show()

    def MLS(self):
        dialog = cross()
        self.dialogs.append(dialog)
        dialog.show()

    #
    def MLRF(self):
        #     #::-------------------------------------------------------------
        #     # This function creates an instance of the Random Forest Regression Algorithm
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()


def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


def data_admit():
    global per_new, x, y, features_list, models, cval
    # Importing Dataset
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
    # Checking for Outliers with InterQuatile Range Detection for Int and Float

    names = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA', 'Chance of Admit']
    features_list = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA',
                     'Research', 'Chance of Admit']

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

    # Checking for Outliers for categoricals

    per_new['University Rating'].unique()
    per_new['Research'].unique()

    x, y = per_new.drop(['Chance of Admit', 'GRE Groups', 'TOEFL Groups'], axis=1), per_new['Chance of Admit']
    models = ['Linear Reg', 'DT Reg', 'RF Reg', 'GB Reg', 'Ada B Reg',
              'ExtraTree Reg', 'K-NeighborsReg', 'SVR']
    score = []
    score.append(np.round(cross_val_score(LinearRegression(), x, y).mean() * 100, 2))
    score.append(np.round(cross_val_score(DecisionTreeRegressor(), x, y).mean() * 100, 2))
    score.append(
        np.round(cross_val_score(RandomForestRegressor(n_estimators=50, random_state=2), x, y).mean() * 100, 2))
    score.append(np.round(cross_val_score(GradientBoostingRegressor(), x, y).mean() * 100, 2))
    score.append(np.round(cross_val_score(AdaBoostRegressor(), x, y).mean() * 100, 2))
    score.append(np.round(cross_val_score(ExtraTreesRegressor(), x, y).mean() * 100, 2))
    score.append(np.round(cross_val_score(KNeighborsRegressor(), x, y).mean() * 100, 2))
    score.append(np.round(cross_val_score(SVR(), x, y).mean() * 100, 2))
    cval = pd.DataFrame(score, index=models, columns=['score'])


if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then call the application
    #::------------------------------------
    data_admit()
    main()
