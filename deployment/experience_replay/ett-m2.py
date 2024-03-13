import torch
import numpy as np
import random
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from submission import submit_predictions_for_test_set
from dataset import get_dataset
from torch.cuda.amp import autocast, GradScaler

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

INPUT_SEQ_LEN = 96
INPUT_DIM = 7
PRED_SEQ_LEN = 96
PRED_DIM = 7
HIDDEN_DIM = 32
NUM_LAYERS = 3
BATCH_SIZE = 64
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics_for_time_series_forecasting(y_test, y_test_pred):
    y_test = y_test.reshape(-1, PRED_SEQ_LEN, PRED_DIM)
    y_test_pred = y_test_pred.reshape(-1, PRED_SEQ_LEN, PRED_DIM)
    mae = np.mean(np.abs(y_test - y_test_pred))
    mse = np.mean((y_test - y_test_pred)**2)
    return mse, mae

class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # 2 for bidirection

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device) # 2 for bidirection
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # 2 for bidirection

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device) # 2 for bidirection
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out


def train_model(model, X_train, y_train, X_valid, y_valid):
    criterion = nn.L1Loss() # Change loss function to Mean Absolute Error (MAE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    valid_data = TensorDataset(torch.tensor(X_valid).float(), torch.tensor(y_valid).float())
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            with autocast():
                output = model(X)
                loss = criterion(output, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            valid_losses = []
            mses = []
            maes = []
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                valid_output = model(X)
                valid_loss = criterion(valid_output, y)
                valid_losses.append(valid_loss.item())
                mse, mae = compute_metrics_for_time_series_forecasting(y.cpu().numpy(), valid_output.cpu().numpy())
                mses.append(mse)
                maes.append(mae)
            print(f"Epoch {epoch+1}, Train Loss: {loss.item()}, Valid Loss: {np.mean(valid_losses)}, MSE: {np.mean(mses)}, MAE: {np.mean(maes)}")

    return model, np.mean(valid_losses)

def predict(model, X):
    model.eval()
    X = torch.tensor(X).float().to(device)
    with torch.no_grad():
        preds = model(X)
    return preds.cpu().numpy()

if __name__ == '__main__':
    # Load training set
    X_train, y_train = get_dataset(flag='train')
    # Load validation set
    X_valid, y_valid = get_dataset(flag='val')

    # define and train the GRU model
    gru_model = BiGRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, PRED_DIM).to(device)
    gru_model, gru_valid_loss = train_model(gru_model, X_train, y_train, X_valid, y_valid)

    # define and train the LSTM model
    lstm_model = BiLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, PRED_DIM).to(device)
    lstm_model, lstm_valid_loss = train_model(lstm_model, X_train, y_train, X_valid, y_valid)

    # combine the predictions of the GRU and LSTM models
    y_valid_pred_gru = predict(gru_model, X_valid)
    y_valid_pred_lstm = predict(lstm_model, X_valid)
    gru_weight = 1 / gru_valid_loss
    lstm_weight = 1 / lstm_valid_loss
    total_weight = gru_weight + lstm_weight
    y_valid_pred = (y_valid_pred_gru * gru_weight + y_valid_pred_lstm * lstm_weight) / total_weight

    # evaluate the performance of this ensemble method on the validation set
    mse, mae = compute_metrics_for_time_series_forecasting(y_valid, y_valid_pred)
    print(f"Final MSE on validation set: {mse}, Final MAE on validation set: {mae}.")

    # Submit predictions on the test set
    X_test, y_test = get_dataset(flag='test')
    y_test_pred_gru = predict(gru_model, X_test)
    y_test_pred_lstm = predict(lstm_model, X_test)
    y_test_pred = (y_test_pred_gru * gru_weight + y_test_pred_lstm * lstm_weight) / total_weight
    submit_predictions_for_test_set(y_test, y_test_pred)