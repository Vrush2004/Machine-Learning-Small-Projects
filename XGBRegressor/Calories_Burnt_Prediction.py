# Importing the dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Data collection and processing
calaries = pd.read_csv(r"C:\Users\Dell\Desktop\Machine Learning Small Projects\XGBRegressor\calories.csv")
print(calaries.head())
exercise_data = pd.read_csv(r"C:\Users\Dell\Desktop\Machine Learning Small Projects\XGBRegressor\exercise.csv")
print(exercise_data.head())

#Combining the two dataframes
calaries_data = pd.concat([exercise_data, calaries['Calories']],axis=1)
print(calaries_data.head())
calaries_data.shape
calaries_data.info()

#Data Analysis
calaries_data.describe()
calaries_data['Gender'] = calaries_data['Gender'].replace({'female': 0, 'male': 1})

#Data Visualization
sns.set()
#Ploting the gender column in count plot
sns.countplot(calaries_data['Gender'])
#Finding the distribution of 'Age' column
sns.displot(calaries_data['Age'])
#Finding the distribution of 'Height' column
sns.displot(calaries_data['Height'])
#Finding the distribution of 'Weight' column
sns.displot(calaries_data['Weight'])

#Finding the Correlation in the dataset
correlation = calaries_data.corr()
#Constructing a heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Reds')

#Seperating features and target
X = calaries_data.drop(columns = ['User_ID','Calories'], axis=1)
Y = calaries_data['Calories']

#Splitting the data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#Model Training: XGBoost Regressor
model = XGBRegressor()
model.fit(X_train, Y_train)

#Evaluation 
#Prediction on test data
test_data_prediction = model.predict(X_test)
print(test_data_prediction)

#Mean Absolute Error
mea = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error: ",mea)



