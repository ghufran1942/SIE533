import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import string
from unidecode import unidecode
import random
import itertools
import csv
import nltk
from nltk.corpus import stopwords
from torchtext.vocab import vocab

PATH = "data"

#### Reading in the data.
content = pd.read_csv(f"{PATH}/content_filtered.csv")
correlations = pd.read_csv(f"{PATH}/correlations.csv")
#sample_submission = pd.read_csv("/kaggle/input/learning-equality-curriculum-recommendations/sample_submission.csv")
topics = pd.read_csv(f"{PATH}/topics_filtered.csv")

#### Create combine function
def combine(correlations, topics, content):
    '''
    - Inputs our three datasets and combines the topic/content information with the topic/content correlations data. 
    - All topic/content information is concatenated to one "features" column, which includes the language, title, description, etc.
    - Output includes the correlations topics information, correlations content information, and a dictionary to convert indices to their
      corresponding topic/content id. 
    '''
    #Drop/combine columns
    content["text"] = content["text"].fillna('')
    content = content.dropna()
    content_combined = content["language"] + " " + content["title"] + " " + content["description"] + " " + content["text"]
    content_combined = pd.DataFrame({"id":content["id"], "features":content_combined})
    print("content_combined", content_combined.shape)

    topics["description"] = topics["description"].fillna('')
    topics = topics.dropna()
    topics_combined = topics["language"] + " " + topics["channel"] + ' ' + topics["title"] + " " + topics["description"]
    topics_combined = pd.DataFrame({"id":topics["id"], "features":topics_combined})
    print("topics_combined", topics_combined.shape)
    
    #Explode correlations rows
    correlations["content_ids"] = correlations["content_ids"].str.split()
    correlations = correlations.explode("content_ids")

    #Merge
    merged = correlations.merge(topics_combined, how="inner", left_on="topic_id", right_on="id")
    print("merged", merged.shape)
    merged = merged.reset_index().merge(content_combined, how="inner", left_on="content_ids", right_on="id", sort=False, suffixes=("_topics", "_content")).sort_values(axis=0, by="index")
    merged = merged.drop(["content_ids", "topic_id"], axis=1)
    print("merged", merged.shape)

    #Split
    corr_topics = merged[['index', 'features_topics']]
    corr_topics.columns = ['id', 'features']
    corr_content = merged[['index', 'features_content']]
    corr_content.columns = ['id', 'features']

    index_to_topic = pd.Series(merged.id_topics.values, index=merged.index).to_dict()
    index_to_content = pd.Series(merged.id_content.values, index=merged.index).to_dict()

    return corr_topics, corr_content, index_to_topic, index_to_content

corr_topics, corr_content, index_to_topic, index_to_content = combine(correlations, topics, content)

# Define the model
class SNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, 1)

    def forward(self, topics, content):
        topics_embed = self.embedding(topics)
        content_embed = self.embedding(content)

        topics_embed = topics_embed.transpose(1, 2)
        content_embed = content_embed.transpose(1, 2)

        topics_avg = self.avg_pool(topics_embed).squeeze()
        content_avg = self.avg_pool(content_embed).squeeze()

        topics_hidden = torch.relu(self.fc1(topics_avg))
        content_hidden = torch.relu(self.fc1(content_avg))

        hidden_concat = torch.cat((topics_hidden, content_hidden), 1)

        output = torch.sigmoid(self.fc2(hidden_concat))

        return output

# Tokenization and vectorization
tokenizer = get_tokenizer('basic_english')

def tokenize_and_vectorize(texts):
    # tokenized = [tokenizer(text) for text in texts]
    tokenized = tokenizer(texts)
    # vec = [[vocab[token] for token in text] for text in tokenized]
    vec = [vocab.stoi[token] for token in tokenized]
    return vec


random.seed(10)
train_indices = random.sample(range(len(corr_content)), round(0.8*len(corr_content))) #80/20 train/test split

#### Split training data so 50% is matching and 50% is not matching
half = round(len(train_indices) / 2)
full = len(train_indices)

train_topics_half = corr_topics.iloc[train_indices[:half], :]
train_content_half = corr_content.iloc[train_indices[:half], :]

#Shift second half so that topics/content are not matching
train_topics_full = corr_topics.iloc[train_indices[half:(full-20)], :] 
train_content_full = corr_content.iloc[train_indices[(half+20):(full)], :] 

train_topics = pd.concat([train_topics_half, train_topics_full]).reset_index().drop("index", axis=1)
train_content = pd.concat([train_content_half, train_content_full]).reset_index().drop("index", axis=1)

#### Repeat for test data
test_topics = corr_topics.drop(train_indices, axis=0)
test_content = corr_content.drop(train_indices, axis=0)

half = round(len(test_topics.features) / 2)
full = len(test_topics.features)

test_topics_half = test_topics.iloc[:half, :]
test_content_half = test_content.iloc[:half, :]

test_topics_full = test_topics.iloc[half:(full - 5), :]
test_content_full = test_content.iloc[(half+5):full, :]

test_topics = pd.concat([test_topics_half, test_topics_full]).reset_index().drop("index", axis=1)
test_content = pd.concat([test_content_half, test_content_full]).reset_index().drop("index", axis=1)

#### Create labels
train_labels = np.array((train_topics.id == train_content.id).astype(int))
test_labels = np.array((test_topics.id == test_content.id).astype(int))

train_topics_vec = [tokenize_and_vectorize(x) for x in train_topics.features]
train_content_vec = [tokenize_and_vectorize(x) for x in train_content.features]

test_topics_vec = [tokenize_and_vectorize(x) for x in test_topics.features]
test_content_vec = [tokenize_and_vectorize(x) for x in test_content.features]

train_topics_torch = [torch.tensor(x) for x in train_topics_vec]
train_content_torch = [torch.tensor(x) for x in train_content_vec]
train_labels_torch = torch.tensor(train_labels, dtype=torch.int32)

test_topics_torch = [torch.tensor(x) for x in test_topics_vec]
test_content_torch = [torch.tensor(x) for x in test_content_vec]
test_labels_torch = torch.tensor(test_labels, dtype=torch.int32)



# Let's build the vocabulary
def yield_tokens(data):
    for text in data:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(pd.concat([corr_topics["features"], corr_content["features"]]).values), specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab['<unk>'])

# Training function
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for topics, content, labels in data_loader:
        topics_vec = tokenize_and_vectorize(topics)
        content_vec = tokenize_and_vectorize(content)

        topics_torch = pad_sequence([torch.tensor(x) for x in topics_vec], batch_first=True).to(device)
        content_torch = pad_sequence([torch.tensor(x) for x in content_vec], batch_first=True).to(device)
        labels_torch = torch.tensor(labels).float().to(device)

        optimizer.zero_grad()
        outputs = model(topics_torch, content_torch)
        loss = criterion(outputs, labels_torch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for topics, content, labels in data_loader:
            topics_vec = tokenize_and_vectorize(topics)
            content_vec = tokenize_and_vectorize(content)

            topics_torch = pad_sequence([torch.tensor(x) for x in topics_vec], batch_first=True).to(device)
            content_torch = pad_sequence([torch.tensor(x) for x in content_vec], batch_first=True).to(device)
            labels_torch = torch.tensor(labels).float().to(device)

            outputs =  model(topics_torch, content_torch)
            loss = criterion(outputs, labels_torch)

        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# Create the model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
embed_dim = 256
hidden_dim = 128
model = SNN(vocab_size, embed_dim, hidden_dim).to(device)

# Training parameters

num_epochs = 5
learning_rate = 0.001
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoaders

train_data = list(zip(train_topics.features, train_content.features, train_labels))
test_data = list(zip(test_topics.features, test_content.features, test_labels))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# Training loop

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

# Evaluate the model

test_loss = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")

# Prediction function

def predict(model, topics, content, device):
    model.eval()
    topics_vec = tokenize_and_vectorize(topics)
    content_vec = tokenize_and_vectorize(content)
    topics_torch = pad_sequence([torch.tensor(x) for x in topics_vec], batch_first=True).to(device)
    content_torch = pad_sequence([torch.tensor(x) for x in content_vec], batch_first=True).to(device)

    with torch.no_grad():
        outputs = model(topics_torch, content_torch)

    return outputs.squeeze().tolist()

# Prediction and submission
THRESHOLD = 0.99994
output_file = "submission.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["topic_id", "content_ids"])
    for i in topics_features.index:
        temp_topic = np.repeat(topics_features[i], len(temp_content))
        temp_content = content_features[content_lang == topics_lang[i]]
        matches = predict(model, temp_topic, temp_content, device)
        matches = [i for i in range(len(matches)) if matches[i] > THRESHOLD]
        matches = " ".join([index_to_content[x] for x in matches])
        writer.writerow([index_to_topic[i], matches])

        writer.writerows([correlations.topic_id, correlations.content_ids]) 
f.close()