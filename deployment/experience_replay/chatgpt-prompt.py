import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import DebertaTokenizer, DebertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import random
from submission import submit_predictions_for_test_set
from torch.cuda.amp import GradScaler, autocast

SEED = 42
LABEL_NUM = 8
LEARNING_RATE = 1e-5  # Adjusted learning rate
EPOCHS = 10  # Increased number of epochs
BATCH_SIZE = 16  # Reduced batch size

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

def compute_metrics_for_classification(y_test, y_test_pred):
    acc = accuracy_score(y_test, y_test_pred) 
    return acc

def train_model(dataloader, model, optimizer, scheduler):
    model.train()
    for epoch in range(EPOCHS):
        for batch in dataloader:
            b_input_ids, b_labels = batch[0].to(device), batch[1].to(device)
            model.zero_grad()
            with autocast():
                outputs = model(b_input_ids, labels=b_labels)
                loss = outputs[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    return model

def predict(model, dataloader):
    model.eval()
    predictions = []
    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.extend(np.argmax(logits, axis=1).flatten())
    return predictions

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    # Process data and store into numpy arrays.
    X = data_df.Sentence.to_numpy()
    y = data_df.Label.to_numpy()

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=SEED)

    # define the tokenizer and model
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large', do_lower_case=True)
    model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large', num_labels=LABEL_NUM)
    model.to(device)

    # tokenize and encode sequences for the training and validation sets
    encoded_data_train = tokenizer.batch_encode_plus(
        X_train.tolist(), 
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding=True, 
        truncation=True,
        max_length=256, 
        return_tensors='pt'
    )

    encoded_data_valid = tokenizer.batch_encode_plus(
        X_valid.tolist(), 
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding=True, 
        truncation=True,
        max_length=256, 
        return_tensors='pt'
    )

    # create Tensor datasets
    dataset_train = TensorDataset(encoded_data_train['input_ids'], torch.tensor(y_train))
    dataset_valid = TensorDataset(encoded_data_valid['input_ids'], torch.tensor(y_valid))

    # create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE)
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE)

    # define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*EPOCHS)

    # train the model
    model = train_model(dataloader_train, model, optimizer, scheduler)

    # evaluate the model on the valid set using compute_metrics_for_classification and print the results
    y_valid_pred = predict(model, dataloader_valid)
    acc = compute_metrics_for_classification(y_valid, y_valid_pred)
    print("final Accuracy on validation set: ", acc)

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    X_submission = submission_df.Sentence.to_numpy()

    # tokenize and encode sequences for the test set
    encoded_data_submission = tokenizer.batch_encode_plus(
        X_submission.tolist(), 
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding=True, 
        truncation=True,
        max_length=256, 
        return_tensors='pt'
    )

    # create Tensor dataset
    dataset_submission = TensorDataset(encoded_data_submission['input_ids'])

    # create dataloader
    dataloader_submission = DataLoader(dataset_submission, batch_size=BATCH_SIZE)

    # make predictions for the test set
    y_submission = predict(model, dataloader_submission)
    submit_predictions_for_test_set(y_submission)