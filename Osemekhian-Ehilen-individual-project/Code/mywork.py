
# SOLOMON
x,y = per_new.drop(['Chance of Admit','GRE Groups','TOEFL Groups'],axis=1), per_new['Chance of Admit']
# The algorithm model that is going to learn from our data to make predictions
# splitting the data 80:20
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)

# Cross Validation to identify best model CONSTANT
models = ['Linear Reg','Decision Tree Reg','Random Forest Reg','Gradient Boosting Reg','Ada Boosting Reg',
          'Extra Tree Reg','K-Neighbors Reg','Support Vector Reg']

mod= [LinearRegression(),DecisionTreeRegressor(max_depth=5),RandomForestRegressor(n_estimators = 50),
      GradientBoostingRegressor(),AdaBoostRegressor(),ExtraTreesRegressor(),KNeighborsRegressor(),SVR()]

def crossval(model_objs,x,y,mod_names_list):
    global df_cv
    score=[]
    for i in model_objs:
        score.append(cross_val_score(i,x,y).mean())
    dic= dict(zip(mod_names_list,score))
    df_cv=pd.DataFrame(list(dic.items()),columns=['Model','Score'])

crossval(mod,x,y,models)
crosval_df=df_cv.copy()
crosval_df.sort_values(by='Score',inplace=True)
sns.barplot(y=crosval_df.Model, x=crosval_df.Score, data=df_cv, orient='h'),plt.tight_layout(),plt.show()


## Linear Regression without SCALING
lr.fit(X_train,y_train)
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
print(f'Linear Regression Model Score: {np.round(score2,6)}%')

## VOTING REGRESSOR

model_1 = LinearRegression()
model_2 = RandomForestRegressor(n_estimators= 80,min_samples_split = 2,min_samples_leaf = 1,max_features = 'sqrt',max_depth = 10,bootstrap = True)
model_3 = DecisionTreeRegressor(max_depth=5,random_state=10)
voting_essemble = VotingRegressor(estimators=[('lr', model_1), ('rf', model_2),('dt',model_3)])
voting_essemble.fit(X_train,y_train)

print('Voting Essemble Score: ',voting_essemble.score(X_test,y_test))




class AdmitGraphs(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a plot to show the relation
    # from each feature in the dataset with the chance of admission
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



class CorrelationPlot(QMainWindow):
    # ;:-----------------------------------------------------------------------
    # This class creates a canvas to draw a correlation plot
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

        res_corr = list_corr_features.corr()
        sns.heatmap(res_corr,annot=True,cbar=False, ax=self.ax1)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


