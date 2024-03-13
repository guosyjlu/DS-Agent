import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from dataset import get_dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from submission import submit_predictions_for_test_set

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

SEQUENCE_LENGTH = 1751
FEATURE_DIM = 3
LABEL_NUM = 4
HIDDEN_SIZE = 50
EPOCHS = 100
LEARNING_RATE = 0.01
STEP_SIZE = 5
GAMMA = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim) # Multiply by 2 because it's bidirectional
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        fc_out = self.fc(gru_out[:, -1, :])
        out = self.softmax(fc_out)
        return out

def compute_metrics_for_time_series_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc

def train_model(X_train, y_train, X_valid, y_valid):
    model = LSTMModel(FEATURE_DIM, HIDDEN_SIZE, LABEL_NUM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    X_train = torch.tensor(X_train).float().to(device)
    y_train = torch.tensor(y_train).long().squeeze().to(device)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model

def predict(model, X):
    model.eval()
    X = torch.tensor(X).float().to(device)
    with torch.no_grad():
        output = model(X)
    preds = output.argmax(dim=1).cpu().numpy()
    return preds

if __name__ == '__main__':
    scaler = StandardScaler()

    X_train, y_train = get_dataset(flag='TRAIN')
    X_valid, y_valid = get_dataset(flag='VALID')

    X_train_shape = X_train.shape
    X_valid_shape = X_valid.shape

    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train_shape)
    X_valid = scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid_shape)

    model = train_model(X_train, y_train, X_valid, y_valid)

    y_valid_pred = predict(model, X_valid)
    acc = compute_metrics_for_time_series_classification(y_valid, y_valid_pred)
    print(f"Final Accuracy on validation set: {acc}.")

    X_test, y_test = get_dataset(flag='TEST')
    X_test_shape = X_test.shape
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test_shape)
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test, y_test_pred)