import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from submission import submit_predictions_for_test_set

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_classification(y_test, y_test_pred):
    roauc = roc_auc_score(y_test, y_test_pred)
    return roauc

def train_model(X_train, y_train, X_valid, y_valid):
    model = RandomForestClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    y_pred = model.predict_proba(X)
    # Select the probabilities for the positive class for each label
    y_pred = np.array([proba[:, 1] for proba in y_pred]).T
    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    X = data_df.drop(['EC1', 'EC2'], axis=1)
    y = data_df[['EC1', 'EC2']].to_numpy()
    
    # apply preprocessing
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_classification and print the results
    y_valid_pred = predict(model, X_valid)
    roauc = compute_metrics_for_classification(y_valid, y_valid_pred)
    print("final area under the ROC curve (AUROC) on validation set: ", roauc)

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    X_test = submission_df.drop(['EC1', 'EC2'], axis=1)
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test_pred)