import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# Fixing the randomness of CUDA.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch Version : {}".format(torch.__version__))
print(DEVICE)

model_save = 'Tweet.pt'
model_name = 'Tweet'
num_epochs = 10
batch_size = 32
learning_rate = 1e-3
num_classes = 3
padding_idx = 0
metadata_each_dim = 10

col=['like_count','quote_count','reply_count','retweet_count','tweet','user_verified','followers_count','following_count','tweet_count','listed_count','label']
label_map = {0:'negative',1:'neutral',2:'positive'}
label_convert = {'negative':0,'neutral':1,'positive':2}

train_data = pd.read_csv('train.csv', names = col, skiprows=1)
test_data = pd.read_csv('test.csv',  names = col, skiprows=1)
val_data = pd.read_csv('valid.csv',  names = col, skiprows=1)

# Replace NaN values with 'NaN'
train_data[['like_count','quote_count','reply_count','retweet_count','followers_count','following_count','tweet_count','listed_count']] = train_data[['like_count','quote_count','reply_count','retweet_count','followers_count','following_count','tweet_count','listed_count']].fillna('0')
train_data.fillna('unknow', inplace=True)

test_data[['like_count','quote_count','reply_count','retweet_count','followers_count','following_count','tweet_count','listed_count']] = test_data[['like_count','quote_count','reply_count','retweet_count','followers_count','following_count','tweet_count','listed_count']].fillna('0')
test_data.fillna('unknow', inplace=True)

val_data[['like_count','quote_count','reply_count','retweet_count','followers_count','following_count','tweet_count','listed_count']] = val_data[['like_count','quote_count','reply_count','retweet_count','followers_count','following_count','tweet_count','listed_count']].fillna('0')
val_data.fillna('unknow', inplace=True)


def textProcess(input_text , max_length = -1):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if max_length == -1:
        tokens = tokenizer(input_text, truncation=True, padding=True)
    else:
        tokens = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_length)
    return tokens


class ArabicDataset(data.Dataset):
    def __init__(self, data_df, tweet, label_onehot, label):
        self.data_df = data_df
        self.tweet = tweet
        self.label_onehot = label_onehot
        self.label = label

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        tweet = self.tweet[idx]
        label_onehot = self.label_onehot[idx]
        label = self.label[idx]
        return tweet, label_onehot, label

train_text = torch.tensor(textProcess(train_data['tweet'].tolist())['input_ids'])
train_label = torch.nn.functional.one_hot(torch.tensor(train_data['label'].replace(label_convert)), num_classes=3).type(torch.float64)
train_dataset = ArabicDataset(train_data, train_text, train_label, torch.tensor(train_data['label'].replace(label_convert)))
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_text = torch.tensor(textProcess(val_data['tweet'].tolist())['input_ids'])
val_label = torch.nn.functional.one_hot(torch.tensor(val_data['label'].replace(label_convert)), num_classes=3).type(torch.float64)

val_dataset = ArabicDataset(val_data, val_text, val_label, torch.tensor(val_data['label'].replace(label_convert)))
val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

test_text = torch.tensor(textProcess(test_data['tweet'].tolist())['input_ids'])
test_label = torch.nn.functional.one_hot(torch.tensor(test_data['label'].replace(label_convert)), num_classes=3).type(torch.float64)

test_dataset = ArabicDataset(test_data, test_text, test_label, torch.tensor(test_data['label'].replace(label_convert)))
test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)  # embedding layer is opetation to singel word
        self.convs = nn.ModuleList([
                      nn.Conv1d(in_channels = embedding_dim, #num input word channels=128 (for each singel word as input)
                        out_channels = n_filters, #num output_feature_map = 128
                        kernel_size = fs)
                      for fs in filter_sizes
                       ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text):
        #text = [batch size, sent len]=(32,263)
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]=(32,263,128)

        embedded = embedded.permute(0, 2, 1)  #The second dimension is the required channel dimension for convolution input.
        #embedded = [batch size, emb dim, sent len]=[32,128,263]

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]=[32,128,263-3or4or5+1]
        #output_size={ ( sent len + 2*padding -filter_size[n] ) / stride } +1

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] #shape[2]=the third dim after coved(sent len - filter_sizes[n] + 1)
        #pooled_n = [batch size, n_filters] (real pooled size equals 1[batch size, n_filters, 1])

        cat = self.dropout(torch.cat(pooled, dim = 1))
        # cat = [batch size, n_filters * len(filter_sizes)]=[32,128*3] (n_filters:filters number)
        # n_filters:filters'number for each size of filters    len(filter_sies):the numbers of covs[3,4,5]=3]
        return self.fc(cat)

class ArabicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx, input_dim, input_dim_metadata, hidden_dim, n_layers, bidirectional):
        super().__init__()

        self.textcnn = TextCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        self.fuse = nn.Linear(output_dim * 1, output_dim)

    def forward(self, text):
        #text = [batch size, sent len]
        #metadata = [batch size, metadata dim]

        text_output = self.textcnn(text)

        fused_output = self.fuse(torch.cat((text_output,), dim=1))

        return fused_output

vocab_size = 30522
embedding_dim = 128
n_filters = 128
filter_sizes = [3,4,5]
output_dim = 3
dropout = 0.4
padding_idx = 0
input_dim = 2 * metadata_each_dim
input_dim_metadata = 9
hidden_dim = 64
n_layers = 1
bidirectional = True

model = ArabicModel(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx, input_dim, input_dim_metadata, hidden_dim, n_layers, bidirectional).to(DEVICE)


# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# weights = torch.tensor([0.6, 0.1, 0.3])
# weights = weights.to(DEVICE)
# criterion = nn.CrossEntropyLoss(weight=weights)
criterion = nn.BCEWithLogitsLoss()

# Record the training process
Train_acc = []
Train_loss = []
Train_macro_f1 = []
Train_micro_f1 = []

Val_acc = []
Val_loss = []
Val_macro_f1 = []
Val_micro_f1 = []

def train(num_epochs, model, train_loader, val_loader, optimizer, criterion, model_save):
    epoch_trained = 0
    train_label_all = []
    train_predict_all = []
    val_label_all = []
    val_predict_all = []
    best_valid_loss = float('inf')

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_trained += 1
        epoch_start_time = time.time()
        # Training
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for tweet, label_onehot, label in train_loader:
            tweet = tweet.to(DEVICE)
            label_onehot = label_onehot.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(tweet)
            loss = criterion(outputs,label_onehot)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, train_predicted = torch.max(outputs, 1)
            train_accuracy += sum(train_predicted == label)
            train_predict_all += train_predicted.tolist()
            train_label_all += label.tolist()


        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
        train_macro_f1 = f1_score(train_label_all, train_predict_all, average='macro')
        train_micro_f1 = f1_score(train_label_all, train_predict_all, average='micro')

        Train_acc.append(train_accuracy.tolist())
        Train_loss.append(train_loss)
        Train_macro_f1.append(train_macro_f1)
        Train_micro_f1.append(train_micro_f1)
# Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for tweet, label_onehot, label in val_loader:
                tweet = tweet.to(DEVICE)
                label_onehot = label_onehot.to(DEVICE)
                label = label.to(DEVICE)

                val_outputs = model(tweet)
                val_loss += criterion(val_outputs, label_onehot).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_accuracy += sum(val_predicted == label)
                val_predict_all += val_predicted.tolist()
                val_label_all += label.tolist()


        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader.dataset)
        val_macro_f1 = f1_score(val_label_all, val_predict_all, average='macro')
        val_micro_f1 = f1_score(val_label_all, val_predict_all, average='micro')

        Val_acc.append(val_accuracy.tolist())
        Val_loss.append(val_loss)
        Val_macro_f1.append(val_macro_f1)
        Val_micro_f1.append(val_micro_f1)

        val_losses = []
        val_accuracies = []
        val_macro_f1s = []
        val_micro_f1s = []

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_macro_f1s.append(val_macro_f1)
        val_micro_f1s.append(val_micro_f1)


        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), model_save)
            print(f'***** Best Result Updated at Epoch {epoch_trained}, Val Loss: {val_loss:.4f} *****')
        # Print the losses and accuracy
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, train Loss: {train_loss:.4f}, train Acc: {train_accuracy:.4f},Train F1 Macro: {train_macro_f1:.4f}, Train F1 Micro: {train_micro_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1 Macro: {val_macro_f1:.4f}, Val F1 Micro: {val_micro_f1:.4f}")

    end_time = time.time()
    training_time = end_time - start_time
    # print(f'Total Training Time: {training_time:.2f}s')


train(num_epochs, model, train_loader, val_loader, optimizer, criterion, model_save)

# Evaluate the model on new data
def test(model, test_loader, model_save):
    model.load_state_dict(torch.load(model_save))
    model.eval()

    test_label_all = []
    test_predict_all = []

    test_loss = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for tweet, label_onehot, label in test_loader:
            tweet = tweet.to(DEVICE)
            label_onehot = label_onehot.to(DEVICE)
            label = label.to(DEVICE)

            test_outputs = model(tweet)
            test_loss += criterion(test_outputs, label_onehot).item()
            _, test_predicted = torch.max(test_outputs, 1)

            test_accuracy += sum(test_predicted == label)
            test_predict_all += test_predicted.tolist()
            test_label_all += label.tolist()

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader.dataset)
    test_macro_f1 = f1_score(test_label_all, test_predict_all, average='macro')
    test_micro_f1 = f1_score(test_label_all, test_predict_all, average='micro')

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test F1 Macro: {test_macro_f1:.4f}, Test F1 Micro: {test_micro_f1:.4f}')

test(model, test_loader, model_save)