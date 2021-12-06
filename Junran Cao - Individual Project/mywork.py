## Junran Cao Codes - Final Project
import pandas as pd
import numpy as np
# I imported these two packages for the ease of displaying my results for this file
# After importing the dataset, my codes are:

print(per.head())
print(per.describe())
per_new= per.copy()

#  Cleaning and Preprocessing
per_new.rename(columns={'Chance of Admit ':'Chance of Admit','LOR ': 'LOR'},inplace=True)
per_new['University Rating']=per_new['University Rating'].astype('category')
per_new = per_new.set_index('Serial No.')
per_new['GRE Groups'] = pd.cut(per_new['GRE Score'],4,labels= [1,2,3,4])
per_new['TOEFL Groups'] = pd.cut(per_new['TOEFL Score'],4, labels = [1,2,3,4])
per_new.head()
per_new.isna().any()

#Checking for outliers - integer and float variables

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
    print(np.where(item>upper) or np.where(item<lower))
    n += 1
# Restricting the data values, making them valid
per_new=per[(per.iloc[:,-1]>=0.01)&(per.iloc[:,-1]<=1)]
per_new=per[(per.iloc[:,1]>=1)&(per.iloc[:,1]<=340)]
per_new=per[(per.iloc[:,2]>=1)&(per.iloc[:,2]<=120)]
per_new=per[(per.iloc[:,3]>=1)&(per.iloc[:,3]<=5)]
per_new=per[(per.iloc[:,4]>=1)&(per.iloc[:,4]<=5)]
per_new=per[(per.iloc[:,5]>=1)&(per.iloc[:,5]<=5)]
per_new=per[(per.iloc[:,6]>=1)&(per.iloc[:,6]<=10)]
per_new=per[(per.iloc[:,7]>=0)&(per.iloc[:,7]<=1)]

#Checking for outliers - categorical variables

per_new['University Rating'].unique()
per_new['Research'].unique()
