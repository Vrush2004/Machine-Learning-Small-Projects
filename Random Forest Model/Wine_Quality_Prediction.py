import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#loading the dataset to a Pandas Dataframe
wine_dataset = pd.read_csv()

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