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

sns.heatmap(per.corr(),annot=True,cmap='summer')
plt.title("Correlation on Admission Features");plt.show()

sns.jointplot(x=per_new['CGPA'],y=per_new['Chance of Admit']);plt.show()
sns.jointplot(x=per_new['CGPA'],y=per_new['Chance of Admit'],hue=per_new['Research']);plt.show()
sns.jointplot(x=per_new['CGPA'],y=per_new['Chance of Admit'],hue=per_new['University Rating']);plt.show()
sns.jointplot(x=per_new['GRE Score'],y=per_new['Chance of Admit'],hue=per_new['Research']);plt.show()
sns.jointplot(x=per_new['GRE Score'],y=per_new['Chance of Admit'],hue=per_new['University Rating']);plt.show()
sns.jointplot(x=per_new['TOEFL Score'],y=per_new['Chance of Admit'],hue=per_new['Research']);plt.show()
sns.jointplot(x=per_new['TOEFL Score'],y=per_new['Chance of Admit'],hue=per_new['University Rating']);plt.show()
sns.boxplot(x=per_new['Chance of Admit'],whis=np.inf); plt.show()
 #--As CGPA and GRE increases, chance of admission increases also.

#American dream selector
z=per_new.describe()
z.iloc[:,-1]
per['Admitted']=per.iloc[:,-1].apply(lambda x:0 if x<per.iloc[:,-1].mean() else 1)

#Decision Tree

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