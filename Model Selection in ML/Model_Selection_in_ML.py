import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Importing the models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# We will be working on the Heart Disease dataset
heart_data = pd.read_csv(r"C:\Users\Dell\Desktop\Model Selection in ML\heart.csv")
heart_data.head()
heart_data.shape
heart_data.isnull().sum()
heart_data['target'].value_counts()         # 1 -> Defective heart   0 -> Healthy heart

#Splitting features and target
X = heart_data.drop(columns = 'target', axis=1)
Y = heart_data['target']
X = np.asarray(X)
Y = np.asarray(Y)

# 1) Model Selection
# Comparing the models with default hyperparameter values using Cross Validation
models = [LogisticRegression(max_iter=1000), SVC(kernel='linear'), KNeighborsClassifier(), RandomForestClassifier(random_state=0)]       # list of models

def compare_models_cross_validation():
    for model in models:
        cv_score = cross_val_score(model, X, Y, cv=5)
        mean_accuracy = sum(cv_score)/len(cv_score)
        mean_accuracy = mean_accuracy*100
        mean_accuracy = round(mean_accuracy, 2)
        print("Cross Validation accuracies for the ",model, "=",cv_score)
        print("Accuracy Score of the ",model, "=",mean_accuracy,"%")

compare_models_cross_validation()

# Inference : For thr heart diseases dataset Random Forest Classifier has the Highest accuracy value with default hyperparameters values

# 2) Comparing the models with different hyperparameter values using GridSearchCV
models_list = [LogisticRegression(max_iter=10000), SVC(), KNeighborsClassifier(), RandomForestClassifier(random_state=0)] 

#creating a dictionary that contains hyperparameter values for the above mentioned models
model_hyperparameters = {
    
    'log_reg_hyperparameters' :{
        'C' : [1,5,10,20]
    },
    'svc_hyperparameters' :{
        'kernel' : ['linear','poly','rbf','sigmoid'],
        'C' : [1,5,10,20]
    },
    'KNN_hyperparameters' :{
        'n_neighbors' : [3,5,10]
    },
    'random_forest_hyperparameters' :{
        'n_estimators' : [10,20,50,100]
    }
    
}

type(model_hyperparameters)
print(model_hyperparameters.keys())

model_keys = list(model_hyperparameters.keys())
print(model_keys)

#Applying GridSearchCV
def ModelSelection(list_of_models, hyperparameter_dictionary):
    result = []
    i = 0
    for model in list_of_models:
        key = model_keys[i]
        param = hyperparameter_dictionary[key]
        i += 1
        print(model)
        print(param)
        
        classifier = GridSearchCV(model, param, cv=5)
        classifier.fit(X,Y)                  # fitting the data 
        result.append({
            'model used' : model,
            'highest score' : classifier.best_score_,
            'best hyperparameters' : classifier.best_params_      
        })
        
    result_dataframe = pd.DataFrame(result, columns= ['model used','highest score','best hyperparameters'])

    return result_dataframe

ModelSelection(models_list, model_hyperparameters)          #Random Forest Classifier with n_estimators = 100 has the highest accuracy

















