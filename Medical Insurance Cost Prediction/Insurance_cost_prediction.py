#Importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Data collection and analysis
insurance_dataset = pd.read_csv(r"C:\Users\Dell\Desktop\Medical Insurance Cost Prediction\insurance.csv")
insurance_dataset.head()
insurance_dataset.shape
insurance_dataset.info()
insurance_dataset.isnull().sum()

#Data Analysis
insurance_dataset.describe()

#distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()

#Gender Column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()

insurance_dataset['sex'].value_counts()

#BMI distribution
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()

#children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children Distribution')
plt.show()

insurance_dataset['children'].value_counts()

#Region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('Region Distribution')
plt.show()

insurance_dataset['region'].value_counts()

#Charges distribution
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()

#Data Preprocessing
#Encoding the categorical features
insurance_dataset.replace({'sex':{'male':0, 'female':1}}, inplace=True)
insurance_dataset.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)
insurance_dataset.replace({'region':{'southwest':0, 'southeast':1, 'northwest':2, 'northeast':3}}, inplace=True)

#Splitting the features and target
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

#Splitting the data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#Model Training : Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Model Evaluation
#prediction on training data
training_data_prediction = regressor.predict(X_train)
#R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print("R squared value : ",r2_train)

#prediction on test data
test_data_prediction = regressor.predict(X_test)
#R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print("R squared value : ",r2_test)

#Building a Predictive System
input_data = (31,1,25,74,0,1,0)

#changing input data to a numpy array 
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = regressor.predict(input_data_reshaped)
print(prediction)

print("The insurance cost is USD " , prediction[0])










