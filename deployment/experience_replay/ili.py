import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from submission import submit_predictions_for_test_set
from dataset import get_dataset

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

INPUT_SEQ_LEN = 36
INPUT_DIM = 7
PRED_SEQ_LEN = 24
PRED_DIM = 7
HIDDEN_DIM = 50
NUM_LAYERS = 3
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics_for_time_series_forecasting(y_test, y_test_pred):
    y_test = y_test.reshape(-1, PRED_SEQ_LEN, PRED_DIM)
    y_test_pred = y_test_pred.reshape(-1, PRED_SEQ_LEN, PRED_DIM)
    mae = np.mean(np.abs(y_test - y_test_pred))
    mse = np.mean((y_test - y_test_pred)**2)
    return mae, mse

class MultiResidualBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MultiResidualBiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.grus = nn.ModuleList([nn.GRU(input_dim if i==0 else hidden_dim*2, hidden_dim, 1, batch_first=True, bidirectional=True) for i in range(num_layers)])
        self.residuals = nn.ModuleList([nn.Linear(input_dim, hidden_dim*2) if i==0 else nn.Identity() for i in range(num_layers)])
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        for i in range(self.num_layers):
            out, h0 = self.grus[i](x if i==0 else out, h0)
            if i!=0:
                out = out + self.residuals[i](out)
        out = self.fc(out[:, -1, :])
        return out

def train_model(X_train, y_train, X_valid, y_valid):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, INPUT_SEQ_LEN * INPUT_DIM))
    X_train = X_train.reshape(-1, INPUT_SEQ_LEN, INPUT_DIM)
    X_valid = scaler.transform(X_valid.reshape(-1, INPUT_SEQ_LEN * INPUT_DIM))
    X_valid = X_valid.reshape(-1, INPUT_SEQ_LEN, INPUT_DIM)

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train.reshape(-1, PRED_SEQ_LEN * PRED_DIM)))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = MultiResidualBiGRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, PRED_SEQ_LEN * PRED_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    grad_scaler = GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with autocast():
                output = model(inputs)
                loss = criterion(output, targets)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        model.eval()
        with torch.no_grad():
            X_valid_tensor = torch.Tensor(X_valid).to(device)
            y_valid_tensor = torch.Tensor(y_valid.reshape(-1, PRED_SEQ_LEN * PRED_DIM)).to(device)
            output = model(X_valid_tensor)
            valid_loss = criterion(output, y_valid_tensor)
        print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {valid_loss.item()}')

        scheduler.step(valid_loss)

    return model, scaler

def predict(model, scaler, X):
    X = scaler.transform(X.reshape(-1, INPUT_SEQ_LEN * INPUT_DIM))
    X = X.reshape(-1, INPUT_SEQ_LEN, INPUT_DIM)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.Tensor(X).to(device)
        preds = model(X_tensor)
    preds = preds.cpu().numpy().reshape(-1, PRED_SEQ_LEN, PRED_DIM)
    return preds

if __name__ == '__main__':
    X_train, y_train = get_dataset(flag='train')
    X_valid, y_valid = get_dataset(flag='val')

    model, scaler = train_model(X_train, y_train, X_valid, y_valid)

    y_valid_pred = predict(model, scaler, X_valid)
    mae, mse = compute_metrics_for_time_series_forecasting(y_valid, y_valid_pred)
    print(f"Final MSE on validation set: {mse}, Final MAE on validation set: {mae}.")

    X_test, y_test = get_dataset(flag='test')
    y_test_pred = predict(model, scaler, X_test)
    submit_predictions_for_test_set(y_test, y_test_pred)