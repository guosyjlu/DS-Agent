import pandas as pd
from sklearn.metrics import mean_squared_log_error
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from submission import submit_predictions_for_test_set
import xgboost as xgb

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_regression(y_test, y_test_pred):
    rmlse = mean_squared_log_error(y_test, y_test_pred, squared=False)
    return rmlse

def train_model(X_train, y_train, X_valid, y_valid, params):
    model = xgb.XGBRegressor(**params, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    y_pred = model.predict(X)
    return y_pred

def handle_missing_values(df):
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    X = data_df.drop(['cost'], axis=1)
    X = handle_missing_values(X)
    y = data_df.cost.to_numpy()
    
    # apply preprocessing
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    model = xgb.XGBRegressor(random_state=SEED)
    param_grid = {'max_depth': [3, 5, 7, 10], 'learning_rate': [0.01, 0.1, 0.2, 0.3]}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_log_error')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    print("Best parameters found: ", best_params)

    model = train_model(X_train, y_train, X_valid, y_valid, best_params)

    # evaluate the model on the valid set using compute_metrics_for_regression and print the results
    y_valid_pred = predict(model, X_valid)
    rmlse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print("final root mean squared log error (RMLSE) on validation set: ", rmlse)

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    X_test = submission_df.drop(['cost'], axis=1)
    X_test = handle_missing_values(X_test)
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test_pred)