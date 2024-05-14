# Failure Cases for DS-Agent

## 1. Development Stage

### 1.1 Unreasonable Plans

**Task Description**

> You are solving this machine learning tasks of regression:
> 
> The dataset presented here (the ELLIPSE corpus) comprises argumentative essays written by 8th-12th grade English Language Learners (ELLs). The essays have been scored according to six analytic measures: cohesion, syntax, vocabulary, phraseology, grammar, and conventions. Each measure represents a component of proficiency in essay writing, with greater scores corresponding to greater proficiency in that measure. The scores range from 1.0 to 5.0 in increments of 0.5. Your task is to predict the score of each of the six measures for the essays given in the test set. The evaluation metric is MCRMSE, mean columnwise root mean squared error.
> 
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a language model to get a good performance.

**DS-Agent with GPT-3.5 gives the plan:**

> 1. Implement the missing functions for training the model and making predictions.
> 
> 2. Modify the code to incorporate the pooling method described in the case.
> 
> 3. Generate variations of the prompt question and use them as additional training data for prompt question augmentation.
> 
> 4. Investigate the use of auxiliary classes by incorporating target labels from external models.
> 
> 5. Experiment with different max lengths for training and inference to find the optimal value.
> 
> 6. Consider using pseudo labels by combining the original labels with pseudo labels.

**Explanation:**

In this plan, DS-Agent does not explicitly give the model design but requires to incorporate a pooling layer. Besides, the data augmentation technique is ambigous and is inappropriate for baseline implementation. Furthermore, how to use pseudo lables is also unclear. As a result, DS-Agent with GPT-3.5 fails in this single run.



### 1.2 Overly Complex Plans

**Task Description**

> You are solving this machine learning tasks of regression:
> 
> The dataset presented here (the ELLIPSE corpus) comprises argumentative essays written by 8th-12th grade English Language Learners (ELLs). The essays have been scored according to six analytic measures: cohesion, syntax, vocabulary, phraseology, grammar, and conventions. Each measure represents a component of proficiency in essay writing, with greater scores corresponding to greater proficiency in that measure. The scores range from 1.0 to 5.0 in increments of 0.5. Your task is to predict the score of each of the six measures for the essays given in the test set. The evaluation metric is MCRMSE, mean columnwise root mean squared error.
> 
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a language model to get a good performance.

**DS-Agent with GPT-3.5 gives the plan:**

> 1. Implement a baseline model using a pre-trained language model (e.g., BERT or RoBERTa) and fine-tune it for the regression task.
> 
> 2. Explore different pooling methods such as mean pooling, LSTM layer pooling, and LSTM sequence pooling to capture important information from the essays.
> 
> 3. Augment the prompt questions by generating variations using techniques like language model-based augmentation.
> 
> 4. Incorporate auxiliary classes based on external models trained on similar data to provide additional information and improve the model's performance.
> 
> 5. Experiment with different max lengths to find the optimal length for the model. Start with a range of 896-1280 during initial training and increase it during pseudo labeling rounds if necessary.
> 
> 6. Consider using pseudo labels to leverage additional data and improve the model's performance. Concatenate the original labels and pseudo labels for training.
> 
> 7. Explore ensembling multiple models, such as deberta-v3-large, deberta-v3-base, OpenAssistant/reward-model-deberta-v3-large-v2, and deberta-large, to further enhance the performance.
> 
> 8. Evaluate the models using the MCRMSE metric and compare the results with the baseline model.
> 
> 9. Submit predictions for the test set using the best-performing model.

**Explanation:**

In this plan, DS-Agent seems to generate a relatively reasonable plan, but regretfully, DS-Agent incorporates too many advanced techniques at one time.  Consequently, Programmer and Debugger fail to implement theses requirements. Thus, DS-Agent with GPT-3.5 fails in this run.



### 1.3 Failure in Debugging

**Task Description**

> You are solving this machine learning tasks of classification:
> 
> The dataset presented here (the Textual Entailment) comprises a series of labeled text pairs. Given two texts, your task is to predict the relationship of the text pair of neutral (0), contradiction (1) or entailment (2). The evaluation metric is accuracy.
> 
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a language model to get a good performance.

**DS-Agent with GPT-3.5 gives the plan:**

> 1. Implement a baseline model using the Longformer architecture.
> 
> 2. Train the baseline model using the provided dataset.
> 
> 3. Evaluate the performance of the baseline model on the validation set using accuracy as the evaluation metric.
> 
> 4. Fine-tune the hyperparameters of the baseline model to improve its performance.

**In the execution step, DS-Agent implements the following script:**

```python
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from torch.optim import AdamW

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text1 = str(self.X[idx][0])
        text2 = str(self.X[idx][1])
        label = self.y[idx]

        encoding = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics_for_classification(y_test, y_test_pred):
    acc = accuracy_score(y_test, y_test_pred) 
    return acc

def train_model(X_train, y_train, X_valid, y_valid):
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    max_len = 512
    batch_size = 8
    epochs = 5

    train_dataset = CustomDataset(X_train, y_train, tokenizer, max_len)
    valid_dataset = CustomDataset(X_valid, y_valid, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=3)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0.0
        y_valid_pred = []

        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                valid_loss += loss.item()
                logits = outputs.logits

            y_pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
            y_valid_pred.extend(y_pred)

        valid_loss /= len(valid_loader)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")

    return model

def predict(model, X):
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    max_len = 512
    batch_size = 8

    dataset = CustomDataset(X, np.zeros(len(X)), tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    y_pred = []

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        y_pred_batch = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_pred.extend(y_pred_batch)

    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    # Process data and store into numpy arrays.
    X = data_df[["text1", "text2"]].to_numpy()
    y = data_df.label.to_numpy()

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_classification and print the results
    y_valid_pred = predict(model, X_valid)
    acc = compute_metrics_for_classification(y_valid, y_valid_pred)
    print("Final Accuracy on validation set:", acc)

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    X_submission = submission_df[["text1", "text2"]].to_numpy()
    y_submission = predict(model, X_submission)
    # submit_predictions_for_test_set(y_submission)
```

**Explanation:**

Although at this case, DS-Agent generates reasonable and relatively simple plans. However, Programmer forgets to submit the final results. Thus, DS-Agent fails in this run.



## 2. Deployment Stage

### 2.1 Shape Mismatch

**Task Description**

> You are solving this data science tasks of multi-label classification: 
> The dataset presented here (the Cirrhosis Outcomes Dataset) comprises a lot of both categorical and numerical features. We have splitted the dataset into three parts of train, valid and test. Your task is to predict the Status_C, Status_D, and Status_CL items, all of which have binary labels with 0 and 1. The evaluation metric is the negative log loss (NLL).
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a multi-label classification model to get a good performance on this task.

**DS-Agent generates the following script:**

```python
import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from submission import submit_predictions_for_test_set

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_classification(y_test, y_test_pred):
    nll = log_loss(y_test, y_test_pred)
    return nll

def train_model(X_train, y_train, X_valid, y_valid):
    model = LogisticRegression(random_state=SEED)
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    y_pred = model.predict_proba(X)
    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    X = data_df.drop(['Status_C', 'Status_D', 'Status_CL'], axis=1)
    y = data_df[['Status_C', 'Status_D', 'Status_CL']].to_numpy()
    
    # identify numerical and categorical features
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
    
    # convert discrete variables to one-hot vector
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)  # No need for additional numerical transformation
    ])
    
    # apply preprocessing
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)
    
    # fit the preprocessor on the training data and transform both training and validation data
    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)

    # define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_classification and print the results
    y_valid_pred = predict(model, X_valid)
    nll = compute_metrics_for_classification(y_valid, y_valid_pred)
    print("final NLL on validation set: ", nll)

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    X_test = submission_df.drop(['Status_C', 'Status_D', 'Status_CL'], axis=1)
    X_test = preprocessor.transform(X_test)
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test_pred)


```

**Error Information:**

> ValueError: y should be a 1d array, got an array of shape (6402, 3) instead.

**Explanation:**

In this case, DS-Agent should only return probabilities of positive labels since the evaluation metric is NLL loss. As a result, this run leads to a shape mismatch error.

### 2.2 Undefined Variables

**Task Description**

> You are solving this machine learning tasks of time series classification: 
> The dataset presented here (the Heartbeat dataset) comprises real-world time series data. We have splitted the dataset into three parts of train, valid and test. The input is a sequence of observed features (INPUT_SEQ_LEN=405, INPUT_DIM=61). Your task is to predict the labels for the given sequence, where the label is in range of {0, 1}. The evaluation metric is accuracy.
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a time series classification model to get a good performance on the given fixed sequences.

**DS-Agent generates the following script:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from dataset import get_dataset
from sklearn.metrics import accuracy_score
from submission import submit_predictions_for_test_set

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

SEQUENCE_LENGTH = 405
FEATURE_DIM = 61
LABEL_NUM = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim * (SEQUENCE_LENGTH - 2), output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def compute_metrics_for_time_series_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc

def train_model(X_train, y_train, X_valid, y_valid):
    model = CNN(FEATURE_DIM, 32, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    X_train, y_train = torch.Tensor(X_train).to(device), torch.Tensor(y_train).long().to(device)
    X_valid, y_valid = torch.Tensor(X_valid).to(device), torch.Tensor(y_valid).long().to(device)

    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train.squeeze(1))
        loss.backward()
        optimizer.step()

    return model

def predict(model, X):
    X = torch.Tensor(X).to(device)
    predictions = model(X)
    preds = torch.argmax(predictions, dim=1).cpu().numpy()
    return preds

if __name__ == '__main__':
    X_train, y_train = get_dataset(flag='TRAIN')
    X_valid, y_valid = get_dataset(flag='VALID')

    model = train_model(X_train, y_train, X_valid, y_valid)
    y_valid_pred = predict(model, X_valid)
    acc = compute_metrics_for_time_series_classification(y_valid, y_valid_pred)
    print(f"Final Accuracy on validation set: {acc}.")

    X_test, y_test = get_dataset(flag='TEST')
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test, y_test_pred)


```

**Error Information:**

> NameError: name 'OUTPUT_DIM' is not defined

**Explanation:**

In this case, DS-Agent uses the variable 'OUTPUT_DIM' without pre-defining it. As a result, this run leads to an undefined variable error.

### 2.3 Incorrect Function Calling

**Task Description**

> You are solving this data science tasks of regression: 
> The dataset presented here (the Concrete Strength Dataset) comprises a lot of numerical features. We have splitted the dataset into three parts of train, valid and test. Your task is to predict the strength item. The evaluation metric is the root mean squared error (RMSE).
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a regression model to get a good performance on this task.

**DS-Agent generates the following script:**

```python
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from submission import submit_predictions_for_test_set
from sklearn.linear_model import LinearRegression

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_regression(y_test, y_test_pred):
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    return rmse

def train_model(X_train, y_train, X_valid, y_valid):
    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Training the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model

def predict(model, X):
    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.transform(X)

    # Predicting
    y_pred = model.predict(X_scaled)

    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    X = data_df.drop(['Strength'], axis=1)
    y = data_df['Strength'].to_numpy()
    
    # apply preprocessing
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_regression and print the results
    y_valid_pred = predict(model, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print("final RMSE on validation set: ", rmse)

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    X_test = submission_df.drop(['Strength'], axis=1)
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test_pred)


```

**Error Information:**

> sklearn.exceptions.NotFittedError: This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

**Explanation:**

In this case, DS-Agent uses the scaler for data transformation without fitting it. As a result, this run leads to a function calling error.

### 2.4 Missing Package Import

**Task Description**

> You are solving this machine learning tasks of regression: 
> The dataset presented here (the Hotel reviews) comprises customer feedback for TripAdvisor hotels. Here, we provide the textual reviews. Your task is to predict the corresponding rating in the range of {1, ..., 5} given the reviews in the test set. The evaluation metric is root mean squared error (RMSE).
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a language model to get a good performance. 

**DS-Agent generates the following script:**

```python
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import torch
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from submission import submit_predictions_for_test_set

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

def compute_metrics_for_regression(y_test, y_test_pred):
    rmse = mean_squared_error(y_test, y_test_pred, squared=False) 
    return rmse

def train_model(X_train, y_train, X_valid, y_valid):
    model = BertRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    for epoch in range(5):
        model.train()
        for i in range(0, len(X_train), 32):
            batch_X = X_train[i:i+32]
            batch_y = y_train[i:i+32]
            encoded_X = [bert_tokenizer.encode(x, add_special_tokens=True) for x in batch_X]
            max_len = min(max(len(x) for x in encoded_X), 512)
            input_ids = torch.tensor([x[:max_len] + [0]*(max_len - len(x)) for x in encoded_X]).to(device)
            attention_mask = (input_ids != 0).float().to(device)
            labels = torch.tensor(batch_y).to(device)
            optimizer.zero_grad()
            output = model(input_ids, attention_mask).squeeze()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)
        model.eval()
        with torch.no_grad():
            y_valid_pred = np.zeros(len(X_valid))
            for i in range(0, len(X_valid), 32):
                batch_X = X_valid[i:i+32]
                encoded_X = [bert_tokenizer.encode(x, add_special_tokens=True) for x in batch_X]
                max_len = min(max(len(x) for x in encoded_X), 512)
                input_ids = torch.tensor([x[:max_len] + [0]*(max_len - len(x)) for x in encoded_X]).to(device)
                attention_mask = (input_ids != 0).float().to(device)
                output = model(input_ids, attention_mask).squeeze().cpu().numpy()
                y_valid_pred[i:i+32] = output
            rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
            print("Epoch {}, Validation RMSE: {}".format(epoch+1, rmse))
    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        y_pred = np.zeros(len(X))
        for i in range(0, len(X), 32):
            batch_X = X[i:i+32]
            encoded_X = [bert_tokenizer.encode(x, add_special_tokens=True) for x in batch_X]
            max_len = min(max(len(x) for x in encoded_X), 512)
            input_ids = torch.tensor([x[:max_len] + [0]*(max_len - len(x)) for x in encoded_X]).to(device)
            attention_mask = (input_ids != 0).float().to(device)
            output = model(input_ids, attention_mask).squeeze().cpu().numpy()
            y_pred[i:i+32] = output
    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    # Process data and store into numpy arrays.
    X = list(data_df.Review.to_numpy())
    y = data_df.Rating.to_numpy()

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_regression and print the results
    y_valid_pred = predict(model, X_valid)
    rmse
```

**Error Information:**

> NameError: name 'nn' is not defined

**Explanation:**

In this case, DS-Agent uses the package torch.nn without import. As a result, this run leads to a missing package import error.

### 2.5  Key Error

**Task Description**

> You are solving this data science tasks of regression: 
> The dataset presented here (the Mohs Hardness Dataset) comprises a lot of numerical features. We have splitted the dataset into three parts of train, valid and test. Your task is to predict the strength item. The evaluation metric is the median absolute error (MedAE).
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a regression model to get a good performance on this task.

**DS-Agent generates the following script:**

```python
import pandas as pd
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import median_absolute_error
from submission import submit_predictions_for_test_set
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_regression(y_test, y_test_pred):
    medae = median_absolute_error(y_test, y_test_pred)
    return medae

def handle_missing_values(df):
    return df.fillna(df.mean())

def handle_categorical_columns(df, categorical_columns):
    for col in categorical_columns:
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_col = pd.DataFrame(encoder.fit_transform(df[col].values.reshape(-1, 1)).toarray(), columns=encoder.get_feature_names_out())
        df = pd.concat([df, encoded_col], axis=1)
        df = df.drop(col, axis=1)
    return df

def train_model(X_train, y_train, X_valid, y_valid):
    rf_model = RandomForestRegressor(random_state=SEED)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_median_absolute_error', return_train_score=False)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    model = RandomForestRegressor(**best_params, random_state=SEED)
    model.fit(X_train, y_train)

    return model

def predict(model, X):
    y_pred = model.predict(X)
    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    X = data_df.drop(['Hardness'], axis=1)
    X = handle_missing_values(X)

    # Handle categorical columns
    categorical_columns = ['Mohs', 'Group']
    X = handle_categorical_columns(X, categorical_columns)

    y = data_df['Hardness'].to_numpy()
    
    # Apply preprocessing
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # Define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # Evaluate the model on the valid set using compute_metrics_for_regression and print the results
    y_valid_pred = predict(model, X_valid)
    medae = compute_metrics_for_regression(y_valid, y_valid_pred)
    print("final MedAE on validation set: ", medae)

    # Submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    X_test = submission_df.drop(['Hardness'], axis=1)
    X_test = handle_missing_values(X_test)
    X_test = handle_categorical_columns(X_test, categorical_columns)
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test_pred)


```

**Error Information:**

> KeyError: 'Mohs'

**Explanation:**

In this case, DS-Agent queries the key 'Mohs' that does not exist. As a result, this run leads to a key error.

### 2.6 Data Type Misalignment

**Task Description**

> You are solving this machine learning tasks of regression: 
> The dataset presented here (the BTC News to Bitcoin Price dataset) comprises a series of BTC news title. Your task is to predict the bitcoin price based on the given BTC news title in the test set. The evaluation metric is root mean squared error (RMSE).
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a language model to get a good performance. 

**DS-Agent generates the following script:**

```python
import pandas as pd
import random
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from submission import submit_predictions_for_test_set

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def compute_metrics_for_regression(y_test, y_test_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return rmse

def train_model(X_train, y_train, X_valid, y_valid):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(5):
        model.train()
        for x_batch, y_batch in zip(X_train, y_train):
            input_ids = torch.tensor(bert_tokenizer.encode(x_batch, add_special_tokens=True)).unsqueeze(0).to(device)
            attention_mask = torch.where(input_ids != 0, torch.tensor(1).to(device), torch.tensor(0).to(device)).unsqueeze(0)
            labels = torch.tensor(y_batch).unsqueeze(0).to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_valid_pred = []
            for x_batch, y_batch in zip(X_valid, y_valid):
                input_ids = torch.tensor(bert_tokenizer.encode(x_batch, add_special_tokens=True)).unsqueeze(0).to(device)
                attention_mask = torch.where(input_ids != 0, torch.tensor(1).to(device), torch.tensor(0).to(device)).unsqueeze(0)
                labels = torch.tensor(y_batch).unsqueeze(0).to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                y_valid_pred.append(outputs.logits.item())

        y_valid_pred = np.array(y_valid_pred)
        rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
        print(f'Epoch {epoch + 1}/{5}, RMSE: {rmse:.4f}')

    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        y_pred = []
        for x_batch in X:
            input_ids = torch.tensor(bert_tokenizer.encode(x_batch, add_special_tokens=True)).unsqueeze(0).to(device)
            attention_mask = torch.where(input_ids != 0, torch.tensor(1).to(device), torch.tensor(0).to(device)).unsqueeze(0)
            outputs = model(input_ids, attention_mask=attention_mask)
            y_pred.append(outputs.logits.item())

    y_pred = np.array(y_pred)
    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')

    # Process data and store into numpy arrays.
    X = list(data_df.top_0.to_numpy())
    y = data_df.price.to_numpy()

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_regression and print the results
    y_valid_pred = predict(model, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(f"final RMSE on validation set: {rmse:.4f}")

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    X_submission = list(submission_df.top_0.to_numpy())
    y_submission = predict(model, X_submission)
    submit_predictions_for_test_set(y_submission)


```

**Error Information:**

> RuntimeError: Found dtype Double but expected Float

**Explanation:**

In this case, DS-Agent calculates two tensors with incompatible data types. As a result, this run leads to a data type misalignment error.

### 2.7 Program Incompleteness

**Task Description**

> You are solving this machine learning tasks of classification: 
> The dataset presented here (the BoolQ dataset) comprises a series of passage-question pairs. Given a passage and a question, your task is to identify whether the question can be inferred from the passage, with 0 as False and 1 as True. The evaluation metric is accuracy.
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a language model to get a good performance. 

**DS-Agent generates the following script:**

```python

import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", add_special_tokens=True)

```

**Error Information:**

> This code is incomplete.

**Explanation:**

In this case, DS-Agent generates incomplete code. As a result, this run fails.
