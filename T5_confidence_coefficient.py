import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5Model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import time
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
from torch.optim.lr_scheduler import StepLR

# Fixing the randomness of CUDA.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch Version : {}".format(torch.__version__))
print(DEVICE)


num_epochs = 10
batch_size = 16  # Reduce batch size
learning_rate = 1e-6
num_classes = 3

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

def textProcess(input_text, max_length=128):
    tokenizer = T5Tokenizer.from_pretrained("UBC-NLP/AraT5-tweet-base")
    tokens = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_length)
    return tokens

class ArabicDataset(torch.utils.data.Dataset):
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
train_label = torch.nn.functional.one_hot(torch.tensor(train_data['label'].replace(label_convert)), num_classes=3).type(torch.float32)
train_dataset = ArabicDataset(train_data, train_text, train_label, torch.tensor(train_data['label'].replace(label_convert)))
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_text = torch.tensor(textProcess(val_data['tweet'].tolist())['input_ids'])
val_label = torch.nn.functional.one_hot(torch.tensor(val_data['label'].replace(label_convert)), num_classes=3).type(torch.float32)
val_dataset = ArabicDataset(val_data, val_text, val_label, torch.tensor(val_data['label'].replace(label_convert)))
val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

test_text = torch.tensor(textProcess(test_data['tweet'].tolist())['input_ids'])
test_label = torch.nn.functional.one_hot(torch.tensor(test_data['label'].replace(label_convert)), num_classes=3).type(torch.float32)
test_dataset = ArabicDataset(test_data, test_text, test_label, torch.tensor(test_data['label'].replace(label_convert)))
test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

# 加载模型和tokenizer
tokenizer = T5Tokenizer.from_pretrained("UBC-NLP/AraT5-tweet-base")
base_model = T5Model.from_pretrained("UBC-NLP/AraT5-tweet-base").to(DEVICE)

class T5Classifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(T5Classifier, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=0.7)  # 添加Dropout层
        self.classifier = nn.Linear(base_model.config.d_model, num_classes)

    def forward(self, input_ids):
        outputs = self.base_model.encoder(input_ids=input_ids)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)  # 应用Dropout
        logits = self.classifier(pooled_output)
        return logits


model = T5Classifier(base_model, num_classes).to(DEVICE)

# 定义损失函数
# 计算类别权重
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_data['label'].replace(label_convert)), y=train_data['label'].replace(label_convert))
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# 定义损失函数，加入类别权重
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

import matplotlib.pyplot as plt
import seaborn as sns

# 训练模型
def train(model, train_loader, optimizer, criterion, num_epochs, val_loader):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        for inputs, labels_onehot, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_f1_macro = f1_score(all_labels, all_preds, average='macro')
        train_f1_micro = f1_score(all_labels, all_preds, average='micro')

        val_loss, val_acc, val_f1_macro, val_f1_micro, val_confidences = evaluate(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"***** Best Result Updated at Epoch {epoch+1}, Val Loss: {val_loss:.4f} *****")

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, train Loss: {train_loss:.4f}, train Acc: {train_acc:.4f}, Train F1 Macro: {train_f1_macro:.4f}, Train F1 Micro: {train_f1_micro:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1 Macro: {val_f1_macro:.4f}, Val F1 Micro: {val_f1_micro:.4f}")

# 在验证集上评估模型
def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_confidences = []
    with torch.no_grad():
        for inputs, labels_onehot, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            confidences = F.softmax(logits, dim=1).max(dim=1)[0]
            all_confidences.extend(confidences.cpu().numpy())

    val_loss = total_loss / len(val_loader)
    val_acc = correct / total
    val_f1_macro = f1_score(all_labels, all_preds, average='macro')
    val_f1_micro = f1_score(all_labels, all_preds, average='micro')

    return val_loss, val_acc, val_f1_macro, val_f1_micro, all_confidences

# 训练模型
train(model, train_loader, optimizer, criterion, num_epochs, val_loader)

# 在测试集上评估模型并获取置信度
test_loss, test_acc, test_f1_macro, test_f1_micro, test_confidences = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1 Macro: {test_f1_macro:.4f}, Test F1 Micro: {test_f1_micro:.4f}")

# 绘制置信度分布图
plt.figure(figsize=(10, 6))
sns.histplot(test_confidences, kde=True, bins=30)
plt.title('Prediction Confidence Distribution')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.show()
