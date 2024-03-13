import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from submission import submit_predictions_for_test_set
from transformers import BertTokenizer, BertModel, AdamW
from torch.nn import Dropout, Linear
from torch.nn.functional import softmax

DIMENSIONS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
DROPOUT_RATE = 0.3

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
base_model = BertModel.from_pretrained('bert-base-uncased')

class CustomDataLoader(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length', stride=256)
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class CustomModel(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base_model = base_model
        self.dropout = Dropout(DROPOUT_RATE)
        self.classifier = Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

model = CustomModel(base_model, len(DIMENSIONS)).to(device)

def compute_metrics_for_regression(y_test, y_test_pred):
    metrics = {}
    for task in DIMENSIONS:
        targets_task = [t[DIMENSIONS.index(task)] for t in y_test]
        pred_task = [l[DIMENSIONS.index(task)] for l in y_test_pred]
        rmse = mean_squared_error(targets_task, pred_task, squared=False)
        metrics[f"rmse_{task}"] = rmse
    rmse = np.mean(list(metrics.values()))
    return rmse

def train_model(X_train, y_train, X_valid, y_valid):
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.MSELoss()

    train_loader = DataLoader(CustomDataLoader(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(3):  # loop over the dataset multiple times
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

def predict(model, X):
    model.eval()
    y_pred = []
    predict_loader = DataLoader(CustomDataLoader(X), batch_size=BATCH_SIZE)

    with torch.no_grad():
        for batch in predict_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            y_pred.extend(outputs.detach().cpu().numpy())
    return np.array(y_pred)

if __name__ == '__main__':

    ellipse_df = pd.read_csv('train.csv', 
                            header=0, names=['text_id', 'full_text', 'Cohesion', 'Syntax', 
                            'Vocabulary', 'Phraseology','Grammar', 'Conventions'], 
                            index_col='text_id')
    ellipse_df = ellipse_df.dropna(axis=0)

    data_df = ellipse_df
    X = list(data_df.full_text.to_numpy())
    y = np.array([data_df.drop(['full_text'], axis=1).iloc[i] for i in range(len(X))])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    model = train_model(X_train, y_train, X_valid, y_valid)

    y_valid_pred = predict(model, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print("final MCRMSE on validation set: ", rmse)

    submission_df = pd.read_csv('test.csv',  header=0, names=['text_id', 'full_text'], index_col='text_id')
    X_submission = list(submission_df.full_text.to_numpy())
    y_submission = predict(model, X_submission)
    submit_predictions_for_test_set(y_submission)