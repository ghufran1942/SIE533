# IMPORT

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import cupy as cp

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
# from cuml.metrics import pairwise_distances

from transformers import AutoModel, AutoTokenizer

# DATA IMPORT

PATH = 'data'
content = pd.read_csv(f'{PATH}/content_filtered.csv')
correlation = pd.read_csv(f'{PATH}/correlations.csv')
topics = pd.read_csv(f'{PATH}/topics_filtered.csv')
submission = pd.read_csv(f'{PATH}/sample_submission.csv')

# SPLIT

# split_train, split_test = train_test_split(correlation,test_size=0.2)
# content_split_train, content_split_test = train_test_split(content,test_size=0.2)

# Model Import
MODEL = 'sentence-transformers/paraphrase-MiniLM-L12-v2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModel.from_pretrained(MODEL)
model.eval()
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# MODEL TRAIN

vecs = []
MAX_LEN = 384
# ELEMENT = content['title']

for _,row in tqdm(content.iterrows(), total=len(content)):
  # print(content['title'][row])
  title = row['title']
  if type(title) is float:
    title = row['description']
  if type(title) is float:
    title = row['text']

  tok = tokenizer(title)
  for k,v in tok.items():
    tok[k] = torch.tensor(v[:MAX_LEN]).to(device).unsqueeze(0)
  with torch.no_grad():
    output = model(**tok)

  vec = output.last_hidden_state.squeeze(0).mean(0).cpu()
  vecs.append(vec)

vecs1 = torch.stack(vecs)

sub_topics_ids = submission['topic_id'].tolist()
_topics = topics.query(f'id in {sub_topics_ids}')

vecs = []
for _,row in tqdm(_topics.iterrows(), total=len(_topics)):
  # print(content['title'][row])
  title = row['title']
  if type(title) is float:
    title = row['description']
  if type(title) is float:
    title = row['This content contains no text.']

  tok = tokenizer(title)
  for k,v in tok.items():
    tok[k] = torch.tensor(v[:MAX_LEN]).to(device).unsqueeze(0)
  with torch.no_grad():
    output = model(**tok)

  vec = output.last_hidden_state.squeeze(0).mean(0).cpu()
  vecs.append(vec)

vecs2 = torch.stack(vecs)

vecs1 = cp.asarray(vecs1)
vecs2 = cp.asarray(vecs2)

# MODEL PREDICTION

vecs1_tensor = vecs1.to('cuda')  # Move vecs1 to the GPU
vecs2_tensor = vecs2.to('cuda')  # Move vecs2 to the GPU

prediction = []

for v2 in tqdm(vecs2_tensor, desc="Processing vecs2"):
    sim = torch.cdist(v2.view(1, -1), vecs1_tensor, p=2)  # Compute pairwise cosine distances
    _, indices = torch.topk(sim, 5, largest=False)  # Find the indices of the top 5 smallest distances
    indices = indices.cpu().numpy()  # Move the indices back to the CPU and convert to a numpy array
    
    predict = " ".join([content.loc[s, 'id'] for s in indices[0]])
    prediction.append(predict)

# split_train['content_ids'] = prediction
# split_train.head()

