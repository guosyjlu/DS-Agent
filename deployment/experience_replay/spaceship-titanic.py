import pandas as pd
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from submission import submit_predictions_for_test_set

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_classification(y_test, y_test_pred):
    accuracy = accuracy_score(y_test, y_test_pred)
    return accuracy

def train_model(X_train, y_train, X_valid, y_valid):
    model = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    y_pred = model.predict(X)
    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    data_df = data_df.drop(['PassengerId', 'Name'], axis=1)
    
    X = data_df.drop(['Transported'], axis=1)
    y = data_df.Transported.to_numpy()
    
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
    
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ])
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)

    model = train_model(X_train, y_train, X_valid, y_valid)

    y_valid_pred = predict(model, X_valid)
    accuracy = compute_metrics_for_classification(y_valid, y_valid_pred)
    print("final Accuracy on validation set: ", accuracy)

    submission_df = pd.read_csv('test.csv')
    submission_df = submission_df.drop(['PassengerId', 'Name'], axis=1)
    X_test = preprocessor.transform(submission_df)
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test_pred)