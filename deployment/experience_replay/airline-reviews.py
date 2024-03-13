import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from submission import submit_predictions_for_test_set

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class BertRegressor(nn.Module):
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        output = self.linear(pooled_output)
        return output

class RobertaRegressor(nn.Module):
    def __init__(self):
        super(RobertaRegressor, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        output = self.linear(pooled_output)
        return output

def compute_metrics_for_regression(y_test, y_test_pred):
    rmse = mean_squared_error(y_test, y_test_pred, squared=False) 
    return rmse

def train_model(X_train, y_train, X_valid, y_valid, model, tokenizer):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(5):
        for i in range(0, len(X_train), 32):
            batch_X = X_train[i:i+32]
            batch_y = y_train[i:i+32]
            encoded_X = [tokenizer.encode(x, add_special_tokens=True) for x in batch_X]
            max_len = min(max(len(x) for x in encoded_X), 512)
            input_ids = torch.tensor([x[:max_len] + [0]*(max_len - len(x)) for x in encoded_X]).to(device)
            attention_mask = (input_ids != 0).float().to(device)
            labels = torch.tensor(batch_y).float().to(device)
            optimizer.zero_grad()
            output = model(input_ids, attention_mask).squeeze()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)
    return model

def predict(model, X, tokenizer):
    model.eval()
    with torch.no_grad():
        encoded_X = [tokenizer.encode(x, add_special_tokens=True) for x in X]
        max_len = min(max(len(x) for x in encoded_X), 512)
        input_ids = torch.tensor([x[:max_len] + [0]*(max_len - len(x)) for x in encoded_X]).to(device)
        attention_mask = (input_ids != 0).float().to(device)
        y_pred = model(input_ids, attention_mask).squeeze().cpu().numpy()
    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    data_df = data_df.dropna(subset=['OverallRating'])
    
    X = list(data_df.ReviewBody.to_numpy())
    y = data_df.OverallRating.to_numpy()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    bert_model = BertRegressor().to(device)
    bert_model = train_model(X_train, y_train, X_valid, y_valid, bert_model, bert_tokenizer)

    roberta_model = RobertaRegressor().to(device)
    roberta_model = train_model(X_train, y_train, X_valid, y_valid, roberta_model, roberta_tokenizer)

    y_valid_pred_bert = predict(bert_model, X_valid, bert_tokenizer)
    y_valid_pred_roberta = predict(roberta_model, X_valid, roberta_tokenizer)
    y_valid_pred = (y_valid_pred_bert + y_valid_pred_roberta) / 2
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print("final RMSE on validation set: ", rmse)

    submission_df = pd.read_csv('test.csv')
    submission_df = submission_df.dropna(subset=['OverallRating'])
    X_submission = list(submission_df.ReviewBody.to_numpy())
    y_submission_bert = predict(bert_model, X_submission, bert_tokenizer)
    y_submission_roberta = predict(roberta_model, X_submission, roberta_tokenizer)
    y_submission = (y_submission_bert + y_submission_roberta) / 2
    submit_predictions_for_test_set(y_submission)