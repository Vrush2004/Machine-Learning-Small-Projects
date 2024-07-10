import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#loading the dataset to a Pandas Dataframe
wine_dataset = pd.read_csv(r"C:\Users\Dell\Desktop\Machine Learning Small Projects\Random Forest Model\winequality-red.csv")

#number of rows and columns in the dataset
wine_dataset.shape

#first 5 rows of the dataset
wine_dataset.head()

#checking for missing values
wine_dataset.isnull().sum()

#Data Analysis and Visualization
wine_dataset.describe()

sns.catplot(x='quality',data=wine_dataset, kind='count')        #number of values for each quality

plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)   #volatile acidity vs quality

plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)   #citric acid vs quality

#Correlation
correlation = wine_dataset.corr()

#constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size:8'},cmap='Blues')

#separate the data and label
X = wine_dataset.drop('quality',axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)

#Train and Test data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=2)
print(Y.shape, Y_train.shape, Y_test.shape)

#Model training- Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)

#Model Evaluation
X_test_prediction = model.predict(X_test)                   #accuracy on test data
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy:',test_data_accuracy)

#Building a Predictive System
input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)

input_data_as_numpy_array = np.asarray(input_data)      #changing the input data to a numpy array

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)           #reshape the data as we are predicting the label for only one instance

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 1):
    print("Good Quality Wine")
else:
    print("Bad Quality Wine")















