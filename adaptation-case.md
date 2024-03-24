# Adaptation Case of DS-Agent

## 1. Development Stage

**Given the task description:**

> You are solving this machine learning tasks of regression: 
> The dataset presented here (the ELLIPSE corpus) comprises argumentative essays written by 8th-12th grade English Language Learners (ELLs). The essays have been scored according to six analytic measures: cohesion, syntax, vocabulary, phraseology, grammar, and conventions. Each measure represents a component of proficiency in essay writing, with greater scores corresponding to greater proficiency in that measure. The scores range from 1.0 to 5.0 in increments of 0.5. Your task is to predict the score of each of the six measures for the essays given in the test set. The evaluation metric is MCRMSE, mean columnwise root mean squared error.
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a language model to get a good performance. 

**DS-Agent retrieves the following solution case from human insight case bank:**

> (1) The overall design of the code is to train a feedback prize effectiveness model using a transformer-based architecture. The code reads in the test data and essay texts, preprocesses the data, tokenizes the input, and creates a dataset for training. It then defines the model architecture, which includes a base transformer model, multi-head attention, and a classification layer. Finally, the code performs inference on the test dataset and saves the predictions.
> 
> (2) The overall model architecture consists of a base transformer model, multi-head attention, and a classification layer. The base transformer model is loaded from a pre-trained model checkpoint. The input text is tokenized using the tokenizer, and the tokenized input is passed through the base transformer model to obtain the encoded representations. The encoded representations are then passed through the multi-head attention layer, which attends to different discourse elements in the text. The output of the multi-head attention layer is passed through a classification layer to obtain the final predictions.
> 
> (3) The important hyperparameters in this code are specified in the `config` dictionary. These hyperparameters include the model directory, maximum sequence length, stride, number of labels, dropout rate, and batch size.
> 
> (4) The optimization objective is to minimize the loss between the predicted labels and the ground truth labels. The loss function used is the cross-entropy loss.
> 
> (5) The advanced machine learning technique used in this code is the transformer-based architecture. Transformers have been shown to be highly effective for natural language processing tasks, including text classification.
> 
> (6) Some important tricks that play a role in high performance include:
> 
> - Tokenizing the input text to capture the important discourse elements.
> - Using multi-head attention to attend to different discourse elements in the text.
> - Using a pre-trained base transformer model to leverage pre-trained representations.
> - Applying dropout regularization to prevent overfitting.
> - Using a data collator with padding to handle variable-length input sequences.
> - Using a sliding window approach for long input sequences to handle memory constraints.
> - Using a custom data loader to efficiently load and process the data.
> - Using a custom data collector to handle padding and batch processing.
> - Using the Accelerate library for distributed training and inference.
> - Using tqdm for progress tracking during training and inference.
> - Using joblib for parallel processing during data loading.
> - Using pickle for serialization and deserialization of Python objects.
> - Using the textblob library for text processing tasks such as separating POS tags.
> - Using the IPython library for interactive computing and debugging.
> - Using the tokenizers library for tokenization of input text.
> - Using the BERTopic library for topic modeling.
> - Using the BERTopic model for topic modeling.
> - Using the glob library for file path matching.
> - Using the pandas library for data manipulation and analysis.
> - Using the numpy library for numerical computations.
> - Using the torch library for deep learning.
> - Using the torch.nn library for building neural network models.
> - Using the torch.utils.data library for handling datasets.
> - Using the transformers library for pre-trained transformer models.
> - Using the DataCollatorWithPadding class for padding and collating data.
> - Using the LayerNorm class for layer normalization.
> - Using the tqdm.auto library for progress tracking.
> - Using the torch.cuda.empty_cache() function to clear GPU memory.
> - Using the gc.collect() function to perform garbage collection.

**Then, DS-Agent generates the following plan:**

> 1. Define a model architecture using a pre-trained transformer model such as BERT or RoBERTa. This model will be used to encode the essays into meaningful representations.
> 
> 2. Implement the training process for the model. The model should be trained to minimize the mean squared error (MSE), which aligns with the evaluation metric MCRMSE. Use the Adam optimizer for training.
> 
> 3. Apply dropout regularization during training to prevent overfitting. Set the dropout rate to a reasonable value, such as 0.1.
> 
> 4. Fill in the predict function to generate predictions from the trained model. The function should take the model and input data as arguments, and return an array of predictions.
> 
> After implementing these steps, run the script to train the model and evaluate its performance on the validation set.

**Explanation:**

The source Kaggle task is a text classification competition, while the target task is a text regression task. By analyzing the target task description, DS-Agent selects several key techniques from the retrieved solution, i.e., pre-trained transformer model and dropout, to develop the experiment plan.



## 2. Deployment Stage

**Given the task description:**

> You are solving this data science tasks of binary classification: 
> The dataset presented here (the Software Defects Dataset) comprises a lot of numerical features. We have splitted the dataset into three parts of train, valid and test. Your task is to predict the defects item, which is a binary label with 0 and 1. The evaluation metric is the area under ROC curve (AUROC).
> We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a binary classification model to get a good performance on this task.

**DS-Agent retrieves the following solution case from human insight case bank:**

```python
import pandas as pd
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from submission import submit_predictions_for_test_set

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_classification(y_test, y_test_pred):
    accuracy = accuracy_score(y_test, y_test_pred)
    return accuracy

def train_model(X_train, y_train, X_valid, y_valid):
    model = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    y_pred = model.predict(X)
    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    data_df = data_df.drop(['PassengerId', 'Name'], axis=1)
    
    X = data_df.drop(['Transported'], axis=1)
    y = data_df.Transported.to_numpy()
    
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
    
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ])
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)

    model = train_model(X_train, y_train, X_valid, y_valid)

    y_valid_pred = predict(model, X_valid)
    accuracy = compute_metrics_for_classification(y_valid, y_valid_pred)
    print("final Accuracy on validation set: ", accuracy)

    submission_df = pd.read_csv('test.csv')
    submission_df = submission_df.drop(['PassengerId', 'Name'], axis=1)
    X_test = preprocessor.transform(submission_df)
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test_pred)
```

**Then, DS-Agent generates the following script for the target task:**

```python
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
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
    model = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    X = data_df.drop(['defects'], axis=1)
    y = data_df.defects.to_numpy()
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    model = train_model(X_train, y_train, X_valid, y_valid)

    y_valid_pred = predict(model, X_valid)
    roauc = compute_metrics_for_classification(y_valid, y_valid_pred)
    print("final area under the ROC curve (AUROC) on validation set: ", roauc)

    submission_df = pd.read_csv('test.csv')
    X_test = submission_df.drop(['defects'], axis=1)
    y_test_pred = predict(model, X_test)
    submit_predictions_for_test_set(y_test_pred)


```

**Explanation:**

The source Kaggle task is a tabular classification evaluated by accuracy, while the target task is a tabular classification evaluated by ROAUC. By analyzing the target task description, DS-Agent adapts the source code for the target task, and modify some blocks, such as data pre-processing and evaluation metric.
