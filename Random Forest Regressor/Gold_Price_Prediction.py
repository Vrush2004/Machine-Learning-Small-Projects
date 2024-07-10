# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Data collection and processing
gold_data = pd.read_csv(r"C:\Users\Dell\Desktop\Random Forest Regressor\gld_price_data.csv")
gold_data.head()
gold_data.tail()
gold_data.shape

gold_data.isnull().sum()        #checking the number of missing values
gold_data.describe()            #getting the statistical measures of the data

gold_data['Date'] = pd.to_datetime(gold_data['Date'], errors='coerce') 

# Correlation
correlation = gold_data.corr()
# Constructing a heatmap to understand the correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8},cmap='Reds')

# correlation values of GLD
print(correlation['GLD'])

#checking the distribution of the GLD Price
sns.displot(gold_data['GLD'],color='green')

#Splitting the features and target
X = gold_data.drop(['Date','GLD'],axis=1)















