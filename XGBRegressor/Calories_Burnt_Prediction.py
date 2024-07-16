# Importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Data collection and processing
calaries = pd.read_csv(r"C:\Users\Dell\Desktop\XGBRegressor\calories.csv")
print(calaries.head())
exercise_data = pd.read_csv(r"C:\Users\Dell\Desktop\XGBRegressor\exercise.csv")
print(exercise_data.head())

#Combining the two dataframes
calaries_data = pd.concat(exercise_data, )