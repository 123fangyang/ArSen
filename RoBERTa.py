# Importing the libraries needed
import requests
import os
import gzip
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import math
import torch
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR
import transformers
from transformers import AutoTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch Version : {}".format(torch.__version__))
print(DEVICE)

col=['tweet id','author id','created_at','like_count','quote_count','reply_count','retweet_count','tweet','user_verified','followers_count','following_count','tweet_count','listed_count','name','user_created_at','description','label']
label_map = {0:'negative',1:'neutral',2:'positive'}
label_convert = {'negative':0,'neutral':1,'positive':2}

df=pd.read_csv('ArSen.csv', names=col, skiprows=1)

class Dataset(data.Dataset):
    """
    Custom dataset class for sentiment analysis using RoBERTa.
    """
    def __init__(self, dataframe, max_len):
        """
        Initializes the dataset object with the dataframe and maximum token length.
        """
        # Initialize the tokenizer for RoBERTa base model.
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.data = dataframe
        # self.tweet id= dataframe.tweet id
        # self.author id = dataframe.author id
        self.created_at = dataframe.created_at
        self.like_count = dataframe.like_count
        self.quote_count = dataframe.quote_count
        self.reply_count = dataframe.reply_count
        self.retweet_count = dataframe.retweet_count
        self.tweet = dataframe.tweet

        # Convert labels to integers and generate one-hot encoded labels.
        self.label = dataframe.label
        self.label_one_hot = torch.nn.functional.one_hot(torch.tensor(dataframe.label, dtype=torch.long), num_classes=3).type(torch.float64)
        self.max_len = max_len

    def __len__(self):
        """
        Returns the total number of reviews in the dataset.
        """
        return len(self.tweet)

    def __getitem__(self, idx):
        """
        Returns the tokenized input IDs, attention masks, token type IDs, and labels for a given index.
        """
        # Preprocess the review text by removing extra spaces.
        text = str(self.tweet[idx])
        text = " ".join(text.split())

        # Tokenize the text and pad/truncate to the maximum length.
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # Return the processed inputs and labels.
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.label[idx], dtype=torch.float),
            'targets_one_hot': self.label_one_hot[idx]
        }

# DataLoader hyperparameters
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 8
VALID_BATCH_SIZE = 2

# The dataset is randomly partitioned into training, testing and validation sets and the corresponding DataLoader is instantiated.

train_size = 0.8
val_test_ratio = 0.5
train_data=df.sample(frac=train_size,random_state=42)
test_data=df.drop(train_data.index).reset_index(drop=True)
val_data=test_data.sample(frac=val_test_ratio,random_state=42)
test_data=test_data.drop(val_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_data.shape))
print("VALIDATION Dataset: {}".format(test_data.shape))
print("TEST Dataset: {}".format(test_data.shape))

training_set = Dataset(train_data, MAX_LEN)
testing_set = Dataset(test_data, MAX_LEN)
validation_set = Dataset(val_data, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

validation_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = data.DataLoader(training_set, **train_params)
testing_loader = data.DataLoader(testing_set, **test_params)
validation_loader = data.DataLoader(validation_set, **validation_params)

class RobertaSA(torch.nn.Module):
    """
    This class defines the architecture of the sentiment analysis model based on RoBERTa.
    It includes a pre-trained RoBERTa model for sequence classification with an additional
    classifier layer on top.
    """
    def __init__(self):
        """
        Initializes the model components and loads the pre-trained RoBERTa model.
        """
        super(RobertaSA, self).__init__()
        # Load the pre-trained RoBERTa model for sequence classification with 3 labels.
        self.roberta = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
        self.pre_classifier = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(3, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Defines the forward pass of the model.

        Parameters:
        - input_ids: Tensor containing the input IDs.
        - attention_mask: Tensor containing the attention mask.
        - token_type_ids: Tensor containing the token type IDs.

        Returns:
        - output: The model's predictions.
        """

        output_1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = self.pre_classifier(hidden_state)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Instantiate the RobertaSA model, and move the model to the GPU

model = RobertaSA()
model.to(DEVICE)

# Define the optimiser and loss function.

LEARNING_RATE = 1e-02
WARMUP_STEPS = 1000  # Number of warm-up steps
TOTAL_STEPS = 10000  # Total number of steps (including warm-up)
MIN_LR = 1e-06  # Minimum learning rate after warm-up

loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

scheduler = LambdaLR(optimizer,
                    lambda step: min((step+1) / WARMUP_STEPS, 1.0) if step < WARMUP_STEPS
                    else max(1.0 - (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS), MIN_LR / LEARNING_RATE))

def calcuate_accuracy(preds, targets):

    """
    Calculate the number of correct shots
    Args:
        preds: Predictive results of the model
        targets: Ground truth labels
    Returns:
        n_correct: the number of correction
    """

    n_correct = (preds==targets).sum().item()
    return n_correct

import time
from sklearn.metrics import f1_score

# Define the list of metrics statistics for the training process, including learning rate, loss, precision and f1 scores on the training and validation sets.
learning_rates = []
tr_loss_list = []
tr_acc_list = []
tr_f1_list = []
val_loss_list = []
val_acc_list = []
val_f1_list = []
train_metrics = {}

# Initialize variables for tracking the best results
best_val_loss = float('inf')
best_epoch = 0
num_epochs = 10
# This function is responsible for training the model for a given epoch. It performs both the training and validation phases within each epoch.
def train(epoch):
    global best_val_loss, best_epoch

    # Define the intermediate variables used for computation during training.
    epoch_start_time = time.time()
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    all_targets = []
    all_predictions = []

    print(f'----------------Epoch {epoch}----------------')
    # Training loop
    model.train()
    for step, data in enumerate(training_loader, 0):
        ids = data['ids'].to(DEVICE, dtype = torch.long)
        mask = data['mask'].to(DEVICE, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(DEVICE, dtype = torch.long)
        targets = data['targets'].to(DEVICE, dtype = torch.long)
        targets_one_hot= data['targets_one_hot'].to(DEVICE, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets_one_hot)
        tr_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(predicted, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        # Log training loss and accuracy every 500 steps
        if step % 500 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct) / nb_tr_examples
            print(f"[Epoch {epoch}] [{step} Step] Training Loss per 500 steps: {loss_step:.6f}")
            print(f"[Epoch {epoch}] [{step} Step] Training Accuracy per 500 steps: {accu_step:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    tr_f1_macro = f1_score(all_targets, all_predictions, average='macro')
    tr_f1_micro = f1_score(all_targets, all_predictions, average='micro')
    tr_f1_weighted = f1_score(all_targets, all_predictions, average='weighted')

    learning_rates.append(optimizer.param_groups[0]['lr'])
    tr_loss_list.append(tr_loss / nb_tr_steps)
    tr_acc_list.append((n_correct) / nb_tr_examples)
    tr_f1_list.append(tr_f1_weighted)

    # Calculate and log epoch-level training loss, accuracy, and F1 score
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct) / nb_tr_examples
    print(f"[Epoch {epoch}] Training Loss Epoch: {epoch_loss:.6f}")
    print(f"[Epoch {epoch}] Training Accuracy Epoch: {epoch_accu:.4f}")
    print(f"[Epoch {epoch}] Training F1 (Weighted) Epoch: {tr_f1_weighted:.4f}")
    print(f"[Epoch {epoch}] Training F1 (Macro) Epoch: {tr_f1_macro:.4f}")
    print(f"[Epoch {epoch}] Training F1 (Micro) Epoch: {tr_f1_micro:.4f}\n")

    # Validation loop
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    nb_val_steps = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for step, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.long)
            targets_one_hot = data['targets_one_hot'].to(DEVICE, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets_one_hot)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            val_correct += calcuate_accuracy(predicted, targets)
            nb_val_steps += 1
            val_total += targets.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_f1_macro = f1_score(all_targets, all_predictions, average='macro')
    val_f1_micro = f1_score(all_targets, all_predictions, average='micro')
    val_f1_weighted = f1_score(all_targets, all_predictions, average='weighted')

    val_loss_list.append(val_loss / nb_val_steps)
    val_acc_list.append((val_correct) / val_total)
    val_f1_list.append(val_f1_weighted)

    # Calculate and log validation loss, accuracy, and F1 score
    val_accuracy = val_correct / val_total
    val_loss /= len(validation_loader)

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    # Check for best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        print(f"***** Best Result Updated at Epoch {epoch}, Val Loss: {val_loss:.4f} *****\n")

    print(f"[Epoch {epoch}] Validation Loss: {val_loss:.6f}")
    print(f"[Epoch {epoch}] Validation Accuracy: {val_accuracy:.4f}")
    print(f"[Epoch {epoch}] Validation F1 (Weighted) Epoch: {val_f1_weighted:.4f}")
    print(f"[Epoch {epoch}] Validation F1 (Macro) Epoch: {val_f1_macro:.4f}")
    print(f"[Epoch {epoch}] Validation F1 (Micro) Epoch: {val_f1_micro:.4f}")
    print(f"[Epoch {epoch}] Running time: {epoch_time:.2f}s\n")


for epoch in range(1, num_epochs + 1):
    train(epoch)
