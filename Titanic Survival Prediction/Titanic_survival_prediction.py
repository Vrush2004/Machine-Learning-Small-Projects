#Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data collection and processing
titanic_data = pd.read_csv(r"C:\Users\Dell\Desktop\Machine Learning Small Projects\Titanic Survival Prediction\titanic.csv")
titanic_data.head()
titanic_data.shape
titanic_data.info()
titanic_data.isnull().sum()

#Handling the missing values
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
#replacing the missing values in age column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
#finding the mode value of embarked column
print(titanic_data['Embarked'].mode())
#replacing the missing values in embarked column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data.isnull().sum()

#Data analysis 
titanic_data.describe()
#finding the number of people survived and not survived
titanic_data['Survived'].value_counts()

#Data Visualization
sns.set()
#making a count plot for survived column
sns.countplot(x='Survived',data=titanic_data)

titanic_data['Sex'].value_counts()
sns.countplot(x='Sex', data=titanic_data)
plt.show()

#number of survivors Gender wise
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.show()

#making a count plot for Pclass column
sns.countplot(x='Pclass',data=titanic_data)

sns.countplot(x='Pclass', hue='Survived', data=titanic_data)

#Encoding the Categories columns
titanic_data.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
titanic_data.head()

#Seperating features and target
X = titanic_data.drop(columns=['Name','Ticket','PassengerId','Survived'],axis=1)
Y = titanic_data['Survived']

#Spliting the data inti training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#Model Training: Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

#Model Evaluation : Accuracy Score
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data :',training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data :',test_data_accuracy)















