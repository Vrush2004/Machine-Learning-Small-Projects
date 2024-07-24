#Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data collection and processing
titanic_data = pd.read_csv(r"C:\Users\Dell\Desktop\Titanic Survival Prediction\titanic.csv")
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
