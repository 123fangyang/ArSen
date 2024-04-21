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

model_save = 'Arabic.pt'
model_name = 'Arabic'
num_epochs = 10
batch_size = 32
learning_rate = 1e-3
num_classes = 3
padding_idx = 0
metadata_each_dim = 10   #textual_context的每个feature大小为10

col=['tweet id','author id','created_at','like_count','quote_count','reply_count','retweet_count','tweet','user_verified','followers_count','following_count','tweet_count','listed_count','name','user_created_at','description','label']
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
    def __init__(self, data_df, tweet, tweet_mask, label_onehot, label ,description, created_at, like_count, quote_count, reply_count, retweet_count,
                  followers_count, following_count, tweet_count, listed_count,  user_verified):
        self.data_df = data_df
        self.tweet = tweet
        self.tweet_mask = tweet_mask
        self.label_onehot = label_onehot
        self.label = label

        self.metadata_text = torch.cat((description.int(),created_at.int()), dim=-1)
        self.metadata_number = torch.cat((torch.tensor(like_count, dtype=torch.float).unsqueeze(1),
                                        torch.tensor(quote_count, dtype=torch.float).unsqueeze(1),
                                        torch.tensor(reply_count, dtype=torch.float).unsqueeze(1),
                                        torch.tensor(retweet_count, dtype=torch.float).unsqueeze(1),
                                        torch.tensor(followers_count, dtype=torch.float).unsqueeze(1),
                                        torch.tensor(following_count, dtype=torch.float).unsqueeze(1),
                                        torch.tensor(tweet_count, dtype=torch.float).unsqueeze(1),
                                        torch.tensor(listed_count, dtype=torch.float).unsqueeze(1),
                                        torch.tensor(user_verified, dtype=torch.float).unsqueeze(1)),dim=-1)


    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        tweet = self.tweet[idx]
        tweet_mask = self.tweet_mask[idx]
        label_onehot = self.label_onehot[idx]
        label = self.label[idx]
        metadata_text = self.metadata_text[idx]
        metadata_number = self.metadata_number[idx]
        return tweet, tweet_mask, label_onehot, label, metadata_text, metadata_number
# Define the data loaders for training and validation
train_text = torch.tensor(textProcess(train_data['tweet'].tolist())['input_ids'])
train_text_masks = torch.tensor(textProcess(train_data['tweet'].tolist())['attention_mask'])
train_label = torch.nn.functional.one_hot(torch.tensor(train_data['label'].replace(label_convert)), num_classes=3).type(torch.float64)
train_description = torch.tensor(textProcess(train_data['description'].tolist(), metadata_each_dim)['input_ids'])
train_created_at = torch.tensor(textProcess(train_data['created_at'].tolist(), metadata_each_dim)['input_ids'])

train_dataset = ArabicDataset(train_data, train_text, train_text_masks, train_label, torch.tensor(train_data['label'].replace(label_convert)),
                                   train_description, train_created_at,
                              train_data['like_count'].tolist(), train_data['quote_count'].tolist(),
                              train_data['reply_count'].tolist(), train_data['retweet_count'].tolist(),
                              train_data['followers_count'].tolist(),train_data['following_count'].tolist(),
                              train_data['tweet_count'].tolist(),train_data['listed_count'].tolist(),
                              train_data['user_verified'].tolist())
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)# 新建了一个数据加载器，shuffle=True：打包batch_size前将数据打乱

val_text = torch.tensor(textProcess(val_data['tweet'].tolist())['input_ids'])
val_text_masks = torch.tensor(textProcess(val_data['tweet'].tolist())['attention_mask'])
val_label = torch.nn.functional.one_hot(torch.tensor(val_data['label'].replace(label_convert)), num_classes=3).type(torch.float64)
val_description = torch.tensor(textProcess(val_data['description'].tolist(), metadata_each_dim)['input_ids'])
val_created_at = torch.tensor(textProcess(val_data['created_at'].tolist(), metadata_each_dim)['input_ids'])

val_dataset = ArabicDataset(val_data, val_text, val_text_masks, val_label, torch.tensor(val_data['label'].replace(label_convert)),
                          val_description, val_created_at,
                          val_data['like_count'].tolist(), val_data['quote_count'].tolist(), # 计数
                              val_data['reply_count'].tolist(), val_data['retweet_count'].tolist(),
                              val_data['followers_count'].tolist(), val_data['following_count'].tolist(),
                              val_data['tweet_count'].tolist(), val_data['listed_count'].tolist(),
                              val_data['user_verified'].tolist())
val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

test_text = torch.tensor(textProcess(test_data['tweet'].tolist())['input_ids'])
test_text_masks = torch.tensor(textProcess(test_data['tweet'].tolist())['attention_mask'])
test_label = torch.nn.functional.one_hot(torch.tensor(test_data['label'].replace(label_convert)), num_classes=3).type(torch.float64)
test_description = torch.tensor(textProcess(val_data['description'].tolist(), metadata_each_dim)['input_ids'])
test_created_at = torch.tensor(textProcess(val_data['created_at'].tolist(), metadata_each_dim)['input_ids'])

test_dataset = ArabicDataset(test_data, test_text, test_text_masks, test_label, torch.tensor(test_data['label'].replace(label_convert)),
                          test_description, test_created_at,
                          test_data['like_count'].tolist(), test_data['quote_count'].tolist(),
                              test_data['reply_count'].tolist(), test_data['retweet_count'].tolist(),
                              test_data['followers_count'].tolist(), test_data['following_count'].tolist(),
                              test_data['tweet_count'].tolist(), test_data['listed_count'].tolist(),
                              test_data['user_verified'].tolist())
test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)  # embedding layer is opetation to singel word
        self.convs = nn.ModuleList([
                      nn.Conv1d(in_channels = embedding_dim, #input word channels=128 (for each singel word as input)
                        out_channels = n_filters, #
                        kernel_size = fs)
                      for fs in filter_sizes
                       ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)
        #embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embedding_dim)  #（20，128）
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=1)  #（128，32，1）
        self.rnn = nn.LSTM(32,            #（32，64，1）
                  hidden_dim,
                  num_layers=n_layers,
                  bidirectional=bidirectional,
                  dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)     #(64*2=128,3)
        self.dropout = nn.Dropout(dropout)
    def forward(self, metadata):
        #metadata = [batch size, metadata dim] = [32,128]

        embedded = self.dropout(self.embedding(metadata))
        #embedded = [batch size, metadata dim, emb dim]

        embedded = torch.reshape(embedded, (metadata.size(0), 128, 1))

        conved = F.relu(self.conv(embedded))  # activation function:ReLu
        #conved = [batch size, n_filters, metadata dim - filter_sizes[n] + 1]

        conved = torch.reshape(conved, (metadata.size(0), 32))

        outputs, (hidden, cell) = self.rnn(conved)
        #outputs = [metadata dim - filter_sizes[n] + 1, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        # hidden = self.dropout(torch.cat((hidden[-1,:], hidden[0,:]), dim = -1))
        #hidden = [batch size, hid dim * num directions]

        return self.fc(outputs)

class ArabicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx, input_dim, input_dim_metadata, hidden_dim, n_layers, bidirectional):
        super().__init__()

        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=output_dim)
        self.textcnn2 = TextCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        self.cnn_bilstm = CNNBiLSTM(input_dim_metadata, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
        self.fuse = nn.Linear(output_dim * 3, output_dim)

    def forward(self, text, text_masks, metadata_text, metadata_number):
        #text = [batch size, sent len]
        #metadata = [batch size, metadata dim]

        text_output = self.bert(text, text_masks)[0]
        metadata_output_text = self.textcnn2(metadata_text)
        metadata_output_number = self.cnn_bilstm(metadata_number)

        fused_output = self.fuse(torch.cat((text_output, metadata_output_text ,metadata_output_number), dim=1))

        return fused_output


vocab_size = 30522
embedding_dim = 128
n_filters = 128
filter_sizes = [3,4,5]
output_dim = 3
dropout = 0.4
padding_idx = 0
input_dim = 2 * metadata_each_dim  #textual的输入维度
input_dim_metadata = 9        #9个数值型的feature
hidden_dim = 64
n_layers = 2
bidirectional = True
#  ArabicModel的实例化（传入的参数）
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
        for tweet, tweet_mask, label_onehot, label, metadata_text, metadata_number in train_loader:
            tweet = tweet.to(DEVICE)
            tweet_mask = tweet_mask.to(DEVICE)
            label_onehot = label_onehot.to(DEVICE)
            label = label.to(DEVICE)
            metadata_text = metadata_text.to(DEVICE)
            metadata_number = metadata_number.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(tweet, tweet_mask, metadata_text, metadata_number)
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
            for tweet, tweet_mask, label_onehot, label, metadata_text, metadata_number in val_loader:
                tweet = tweet.to(DEVICE)
                tweet_mask = tweet_mask.to(DEVICE)
                label_onehot = label_onehot.to(DEVICE)
                label = label.to(DEVICE)
                metadata_text = metadata_text.to(DEVICE)
                metadata_number = metadata_number.to(DEVICE)

                val_outputs = model(tweet, tweet_mask, metadata_text, metadata_number)
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
        for tweet, tweet_mask, label_onehot, label, metadata_text, metadata_number in test_loader:
            tweet = tweet.to(DEVICE)
            tweet_mask = tweet_mask.to(DEVICE)
            label_onehot = label_onehot.to(DEVICE)
            label = label.to(DEVICE)
            metadata_text = metadata_text.to(DEVICE)
            metadata_number = metadata_number.to(DEVICE)

            test_outputs = model(tweet, tweet_mask, metadata_text, metadata_number)
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