import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaTokenizer, DebertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from submission import submit_predictions_for_test_set
from torch.cuda.amp import GradScaler, autocast

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextPairDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(text[0], text[1], add_special_tokens=True, padding='max_length', max_length=512, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        if self.labels is not None:
            label = self.labels[idx]
            return input_ids, attention_mask, label
        return input_ids, attention_mask

def compute_metrics_for_classification(y_test, y_test_pred):
    acc = accuracy_score(y_test, y_test_pred) 
    return acc

def save_best_model(model, best_acc, acc, model_path='best_model.pt'):
    if acc > best_acc:
        torch.save(model.state_dict(), model_path)
        print(f'Saved the new best model with acc: {acc}')
        best_acc = acc
    return best_acc

def train_model(X_train, y_train, X_valid, y_valid):
    train_dataset = TextPairDataset(X_train, y_train)
    valid_dataset = TextPairDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=3).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()

    best_acc = 0
    for epoch in range(5):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        y_valid_pred = predict(model, X_valid)
        acc = compute_metrics_for_classification(y_valid, y_valid_pred)
        best_acc = save_best_model(model, best_acc, acc)

    model.load_state_dict(torch.load('best_model.pt'))
    return model

def predict(model, X):
    dataset = TextPairDataset(X)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = [item.to(device) for item in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds.append(outputs.logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    X = data_df[["text1", "text2"]].to_numpy()
    y = data_df.label.to_numpy()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    model = train_model(X_train, y_train, X_valid, y_valid)

    y_valid_pred = predict(model, X_valid)
    acc = compute_metrics_for_classification(y_valid, y_valid_pred)
    print("final Accuracy on validation set: ", acc)

    submission_df = pd.read_csv('test.csv')
    X_submission = submission_df[["text1", "text2"]].to_numpy()
    y_submission = predict(model, X_submission)
    submit_predictions_for_test_set(y_submission)