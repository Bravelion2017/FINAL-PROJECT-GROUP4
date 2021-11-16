""" Introduction to Data Mining Project
    Group-4
    Names- Ricardo, Justin, Jeremiah, Osemekhian
"""

######################################################################
#---
import sys

from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
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
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import export_graphviz
import os
from pydotplus import graph_from_dot_data
import webbrowser
#---
font_size_window = 'font-size:15px'
#---

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

        self.Title = 'Heatmap Plot'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  Creates the canvas and elements of the canvas
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Heatmap Plot for Features')
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

        self.btnExecute = QPushButton("Creat Heatmap")
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


        self.groupBox2 = QGroupBox('Heatmap Plot')
        self.groupBox2Layout= QVBoxLayout()
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
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[3]]],axis=1)
        if self.feature4.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[6]]],axis=1)

        if self.feature7.isChecked():
            list_corr_features = pd.concat([list_corr_features, per_new[features_list[7]]],axis=1)


        vsticks = ["dummy"]
        vsticks1 = list(list_corr_features.columns)
        vsticks1 = vsticks + vsticks1
        res_corr = list_corr_features.corr()
        self.ax1.matshow(res_corr, cmap= plt.cm.get_cmap('Blues', 14))
        self.fig.colorbar(self.ax1.matshow(res_corr, cmap= plt.cm.get_cmap('Blues', 14)))
        self.ax1.set_yticklabels(vsticks1)
        self.ax1.set_xticklabels(vsticks1,rotation = 90)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


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
        # Crate a canvas with the layout to draw a jointplot
        # The layout sets all the elements and manage the changes
        # made on the canvas
        #::--------------------------------------------------------
        super(AdmitGraphs, self).__init__()

        self.Title = "Features vrs Admit Chance"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
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
        # The purpose of the method es to draw a dot graph using the
        # score of happiness and the feature chosen the canvas
        #::--------------------------------------------------------
        colors=["b", "r", "g", "y", "k", "c"]
        self.ax1.clear()
        cat1 = self.dropdown1.currentText()

        X_1 = per_new["Chance of Admit"]
        y_1 = per_new[cat1]


        self.ax1.scatter(X_1,y_1)

        if self.checkbox1.isChecked():

            b, m = polyfit(X_1, y_1, 1)

            self.ax1.plot(X_1, b + m * X_1, '-', color="green")

        vtitle = "Chance of Admission vrs "+ cat1
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel("Chance of Admission")
        self.ax1.set_ylabel(cat1)
        self.ax1.grid(True)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()



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
    #;;----------------------------------
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
        # Creates the actions for the EDA Analysis item
        # Initial Assesment : Histogram about the level of happiness in 2017
        # Happiness Final : Presents the correlation between the index of happiness and a feature from the datasets.
        # Correlation Plot : Correlation plot using all the dims in the datasets
        #::----------------------------------------

        EDA1Button = QAction(QIcon('analysis.png'),'Distribution', self)
        EDA1Button.setStatusTip('Distribution of Chance of Admission')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        EDA2Button = QAction(QIcon('analysis.png'), 'Scatter Plots', self)
        EDA2Button.setStatusTip('Twain features relationship')
        EDA2Button.triggered.connect(self.EDA2)
        EDAMenu.addAction(EDA2Button)

        EDA4Button = QAction(QIcon('analysis.png'), 'Heatmap Plot', self)
        EDA4Button.setStatusTip('Features Correlation Plot')
        EDA4Button.triggered.connect(self.EDA4)
        EDAMenu.addAction(EDA4Button)


        #::--------------------------------------------------
        # ML Models for prediction
        # There are two models
        #       Decision Tree
        #       Random Forest
        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------

        # MLModel1Button =  QAction(QIcon(), 'Decision Tree Gini', self)
        # MLModel1Button.setStatusTip('ML Algorithm with Gini ')
        # MLModel1Button.triggered.connect(self.MLDT)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------

        # MLModel2Button = QAction(QIcon(), 'Random Forest Regression', self)
        # MLModel2Button.setStatusTip('Random Forest Regression ')
        # MLModel2Button.triggered.connect(self.MLRF)
        #
        # MLModelMenu.addAction(MLModel1Button)
        # MLModelMenu.addAction(MLModel2Button)
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
        dialog.m.ax.hist(y,bins=10, facecolor='blue', alpha=0.5)
        dialog.m.ax.set_title('Frequency of Chance of Admission')
        dialog.m.ax.set_xlabel("Chance of Admission")
        dialog.m.ax.set_ylabel("Count of Students")
        dialog.m.ax.grid(True)
        dialog.m.draw()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        #::---------------------------------------------------------
        # This function creates an instance of HappinessGraphs class
        # This class creates a graph using the features in the dataset
        # happiness vrs the score of happiness
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


    # def MLDT(self):
    #     #::-----------------------------------------------------------
    #     # This function creates an instance of the DecisionTree class
    #     # This class presents a dashboard for a Decision Tree Algorithm
    #     # using the happiness dataset
    #     #::-----------------------------------------------------------
    #     dialog = DecisionTree()
    #     self.dialogs.append(dialog)
    #     dialog.show()
    #
    # def MLRF(self):
    #     #::-------------------------------------------------------------
    #     # This function creates an instance of the Random Forest Classifier Algorithm
    #     # using the happiness dataset
    #     #::-------------------------------------------------------------
    #     dialog = RandomForest()
    #     self.dialogs.append(dialog)
    #     dialog.show()

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
    global per_new, x, y, features_list
    # Importing Dataset
    per = pd.read_csv("Admission_Predict.csv")
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
    features_list=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA',
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

if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then call the application
    #::------------------------------------
    data_admit()
    main()