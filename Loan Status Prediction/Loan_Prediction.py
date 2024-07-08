# Importing the dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data collection and processing
loan_dataset = pd.read_csv(r"C:\Users\Dell\Desktop\Loan Status Prediction\loan data.csv")
type(loan_dataset)
print(loan_dataset.head())
loan_dataset.shape
loan_dataset.describe()

loan_dataset.isnull().sum()
loan_dataset = loan_dataset.dropna()

#Label encoding 
loan_dataset.replace({'Loan_Status':{'N':0, 'Y':1}}, inplace = True)

#Dependent column values
loan_dataset['Dependents'].value_counts()
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

#Data visualization
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)    #Education & loan status
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)    #marital status and loan status

# Convert catogorical columns to numerical values
loan_dataset.replace({'Married':{'No':0, 'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':'1'},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

#Seperating the data and label
x = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y = loan_dataset['Loan_Status']

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

#Training the model - SVM model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)               #training the SVM model

#Model evaluation
x_train_prediction = classifier.predict(x_train)        #accuracy score on training data
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy on training data: ",training_data_accuracy)

x_test_prediction = classifier.predict(x_test)        #accuracy score on test data
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy on test data: ",test_data_accuracy)


















