#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing the dataset
dataset=pd.read_csv('Loan payments data.csv')

#categorical variables (converting string to numericals)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
dataset['loan_status']=le.fit_transform(dataset['loan_status'])
dataset['past_due_days']=le.fit_transform(dataset['past_due_days'])
dataset['education']=le.fit_transform(dataset['education'])
dataset['Gender']=le.fit_transform(dataset['Gender'])

#finding which feature/s matter most
sns.barplot(x="Gender",y="loan_status", data=dataset);
sns.barplot(x="age",y="loan_status", data=dataset);
sns.barplot(x="education",y="loan_status", data=dataset);
sns.countplot(x="Gender",data=dataset); #says that number of males are more than females

#removing the features that do not affect the prediciton
data1=dataset 
data1.drop("Loan_ID",axis=1,inplace=True)
data1.drop("effective_date",axis=1,inplace=True)
data1.drop("due_date",axis=1,inplace=True)
data1.drop("paid_off_time",axis=1,inplace=True)
X=data1.pop('loan_status') #(label) independent feature
#data1.head(5)

from sklearn.model_selection import train_test_split
dataset_train ,dataset_test ,X_train ,X_test = train_test_split(data1, X, test_size = 0.2, random_state = 42) 

#applying logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)

#fitting the training sets into the classifier
classifier.fit(dataset_train,X_train)

#predicting the training and testing score 
score_train = classifier.score(dataset_train, X_train)
print("Training score: ",score_train)
score_test = classifier.score(dataset_test, X_test)
print("Testing score: ",score_test)

#predicting the values by applying the classifier on dataset_test
pred = classifier.predict(dataset_test)

