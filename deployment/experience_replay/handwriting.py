import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader
from dataset import get_dataset
from sklearn.metrics import accuracy_score
from submission import submit_predictions_for_test_set

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

SEQUENCE_LENGTH = 152
FEATURE_DIM = 3
LABEL_NUM = 26
HIDDEN_DIM = 150
N_LAYERS = 3
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
WINDOW_SIZE = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics_for_time_series_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc

class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out + self.skip(x)
        out = self.fc(out[:, -1, :])
        return out

def sliding_window(X, y, window_size):
    X_new = []
    y_new = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1] - window_size + 1):
            X_new.append(X[i, j:j+window_size, :])
            y_new.append(y[i])
    return np.array(X_new), np.array(y_new)

def train_model(train_loader, valid_loader):
    model = ResidualLSTM(FEATURE_DIM, HIDDEN_DIM, LABEL_NUM, N_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for X_valid, y_valid in valid_loader:
                X_valid, y_valid = X_valid.to(device), y_valid.to(device)
                valid_predictions = model(X_valid)
                valid_loss = criterion(valid_predictions, y_valid.view(-1))
                total_valid_loss += valid_loss.item()

        print(f'Epoch: {epoch+1:02}, Train Loss: {loss:.3f}, Val Loss: {total_valid_loss/len(valid_loader):.3f}')
        torch.cuda.empty_cache()

    return model

def predict(model, loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X in loader:
            X = X[0].to(device)
            batch_predictions = model(X)
            predictions.extend(batch_predictions.argmax(dim=1).cpu().numpy())
    return np.array(predictions)

if __name__ == '__main__':
    # Load training set
    X_train, y_train = get_dataset(flag='TRAIN')
    # Load validation set
    X_valid, y_valid = get_dataset(flag='VALID')

    # Apply sliding window to the sequences
    X_train, y_train = sliding_window(X_train, y_train, WINDOW_SIZE)
    X_valid, y_valid = sliding_window(X_valid, y_valid, WINDOW_SIZE)

    # Create DataLoaders
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    valid_data = TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.long))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

    # define and train the model
    model = train_model(train_loader, valid_loader)

    # evaluate the model on the valid set using compute_metrics_for_time_series_forecasting and print the results
    y_valid_pred = predict(model, valid_loader)
    acc = compute_metrics_for_time_series_classification(y_valid, y_valid_pred)
    print(f"Final Accuracy on validation set: {acc}.")

    # Submit predictions on the test set
    X_test, y_test = get_dataset(flag='TEST')
    X_test, y_test = sliding_window(X_test, y_test, WINDOW_SIZE)
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    y_test_pred = predict(model, test_loader)
    submit_predictions_for_test_set(y_test, y_test_pred)