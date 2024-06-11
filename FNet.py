from transformers import FNetForSequenceClassification, FNetTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import f1_score
import time
import numpy as np

# Fixing the randomness of CUDA.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch Version : {}".format(torch.__version__))
print(DEVICE)

# Read data
col = ['tweet id', 'author id', 'created_at', 'like_count', 'quote_count', 'reply_count', 'retweet_count', 'tweet',
       'user_verified', 'followers_count', 'following_count', 'tweet_count', 'listed_count', 'name', 'user_created_at',
       'description', 'label']

train_data = pd.read_csv('train.csv', names=col, skiprows=1)
test_data = pd.read_csv('test.csv', names=col, skiprows=1)
val_data = pd.read_csv('valid.csv', names=col, skiprows=1)


# Custom Dataset Class
class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet = self.data.iloc[idx]['tweet']
        label = int(self.data.iloc[idx]['label'])

        metadata_number = self.data.iloc[idx][
            ['like_count', 'quote_count', 'reply_count', 'retweet_count', 'followers_count', 'following_count',
             'tweet_count', 'listed_count']].values.astype(np.float32)

        encoding = self.tokenizer.encode_plus(
            tweet,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return input_ids, attention_mask, label, metadata_number


# Load pretrained FNet model and tokenizer
model_name = 'google/fnet-base'
tokenizer = FNetTokenizer.from_pretrained(model_name)
model = FNetForSequenceClassification.from_pretrained(model_name, num_labels=3).to(DEVICE)

# Data loaders
BATCH_SIZE = 16
MAX_LEN = 128

train_dataset = TweetDataset(train_data, tokenizer, MAX_LEN)
val_dataset = TweetDataset(val_data, tokenizer, MAX_LEN)
test_dataset = TweetDataset(test_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Record the training process
Train_acc = []  # Accuracy
Train_loss = []
Train_macro_f1 = []
Train_micro_f1 = []
Train_f1_ave = []

Val_acc = []
Val_loss = []
Val_macro_f1 = []
Val_micro_f1 = []
Val_f1_ave = []


# Training and evaluation functions
def train(num_epochs, model, train_loader, val_loader, optimizer, criterion, model_save):
    epoch_trained = 0
    best_valid_loss = float('inf')

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_trained += 1
        epoch_start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_label_all = []
        train_predict_all = []

        for input_ids, attention_mask, label, metadata_number in train_loader:
            input_ids = input_ids.to(DEVICE)
            label = label.to(DEVICE)
            metadata_number = metadata_number.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, train_predicted = torch.max(outputs.logits, 1)
            train_accuracy += sum(train_predicted == label)
            train_predict_all += train_predicted.tolist()
            train_label_all += label.tolist()

        train_loss /= len(train_loader)
        train_accuracy = train_accuracy.double() / len(train_loader.dataset)
        train_macro_f1 = f1_score(train_label_all, train_predict_all, average='macro')
        train_micro_f1 = f1_score(train_label_all, train_predict_all, average='micro')
        train_f1_ave = f1_score(train_label_all, train_predict_all, average='weighted')

        Train_acc.append(train_accuracy.item())
        Train_loss.append(train_loss)
        Train_macro_f1.append(train_macro_f1)
        Train_micro_f1.append(train_micro_f1)
        Train_f1_ave.append(train_f1_ave)

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_label_all = []
        val_predict_all = []

        with torch.no_grad():
            for input_ids, attention_mask, label, metadata_number in val_loader:
                input_ids = input_ids.to(DEVICE)
                label = label.to(DEVICE)
                metadata_number = metadata_number.to(DEVICE)

                val_outputs = model(input_ids)
                val_loss += criterion(val_outputs.logits, label).item()
                _, val_predicted = torch.max(val_outputs.logits, 1)
                val_accuracy += sum(val_predicted == label)
                val_predict_all += val_predicted.tolist()
                val_label_all += label.tolist()

        val_loss /= len(val_loader)
        val_accuracy = val_accuracy.double() / len(val_loader.dataset)
        val_macro_f1 = f1_score(val_label_all, val_predict_all, average='macro')
        val_micro_f1 = f1_score(val_label_all, val_predict_all, average='micro')
        val_f1_ave = f1_score(val_label_all, val_predict_all, average='weighted')

        Val_acc.append(val_accuracy.item())
        Val_loss.append(val_loss)
        Val_macro_f1.append(val_macro_f1)
        Val_micro_f1.append(val_micro_f1)
        Val_f1_ave.append(val_f1_ave)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), model_save)
            print(f'***** Best Result Updated at Epoch {epoch_trained}, Val Loss: {val_loss:.4f} *****')

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1 Average: {train_f1_ave:.4f}, Train F1 Macro: {train_macro_f1:.4f}, Train F1 Micro: {train_micro_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1 Macro: {val_macro_f1:.4f}, Val F1 Micro: {val_micro_f1:.4f}, Val F1 Average: {val_f1_ave:.4f}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f'Total Training Time: {training_time:.2f}s')


# Model save path
model_save = 'fnet_model.pt'

# Number of epochs
num_epochs = 10

# Train the model
train(num_epochs, model, train_loader, val_loader, optimizer, criterion, model_save)


# Testing function
def test(model, test_loader, model_save):
    model.load_state_dict(torch.load(model_save))
    model.eval()

    test_label_all = []
    test_predict_all = []

    test_loss = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, label, metadata_number in test_loader:
            input_ids = input_ids.to(DEVICE)
            label = label.to(DEVICE)
            metadata_number = metadata_number.to(DEVICE)

            test_outputs = model(input_ids)
            test_loss += criterion(test_outputs.logits, label).item()

            _, test_predicted = torch.max(test_outputs.logits, 1)
            test_accuracy += sum(test_predicted == label)
            test_predict_all += test_predicted.tolist()
            test_label_all += label.tolist()

    test_loss /= len(test_loader)
    test_accuracy = test_accuracy.double() / len(test_loader.dataset)
    test_macro_f1 = f1_score(test_label_all, test_predict_all, average='macro')
    test_micro_f1 = f1_score(test_label_all, test_predict_all, average='micro')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test F1 Macro: {test_macro_f1:.4f}, Test F1 Micro: {test_micro_f1:.4f}')

test(model, test_loader, model_save)

