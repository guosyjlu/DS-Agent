import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from submission import submit_predictions_for_test_set

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_regression(y_test, y_test_pred):
    mae = mean_absolute_error(y_test, y_test_pred)
    return mae

def train_model(X_train, y_train, X_valid, y_valid):
    model = RandomForestRegressor(random_state=SEED)
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    y_pred = model.predict(X)
    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    X = data_df.drop(['yield'], axis=1)
    y = data_df['yield'].to_numpy()
    
    # apply preprocessing
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_regression and print the results
    y_valid_pred = predict(model, X_valid)
    mae = compute_metrics_for_regression(y_valid, y_valid_pred)
    print("final MAE on validation set: ", mae)

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    X_test = submission_df.drop(['yield'], axis=1)
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test_pred)