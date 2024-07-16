import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data collection and preprocessing
raw_mail_data = pd.read_csv(r"C:\Users\Dell\Desktop\Machine Learning Small Projects\Spam Mail Prediction\mail_data.csv")
print(raw_mail_data.head())

#replace null value with null strings
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

#checking the number of rows and columns in dataframe
mail_data.shape

#Label Encoding : Spam mail = 0, Ham mail = 1
mail_data.loc[mail_data['Category'] == 'spam','Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham','Category',] = 1

#Seperating the data as texts and label
X = mail_data['Message']
Y = mail_data['Category']

#Spliting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(X.shape, X_train.shape, X_test.shape)

#Feature Extraction:
#Transform the text data to feature vectors that can be used as input to the Logistic Regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#Convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#Train the model : Logistic Regression
model = LogisticRegression()
model.fit(X_train_features, Y_train)

#Evaluating the model
#Prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print("Accuracy on training data: ",accuracy_on_training_data)

#Prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print("Accuracy on test data: ",accuracy_on_test_data)

#Building a Predictive System
input_mail = []

#convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

#Making prediction
prediction = model.predict(input_data_features)
print(prediction)

if prediction[0] == 1:
    print("Ham mail")
else :
    print("Spam Mail")



