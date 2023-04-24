# %%
#### Import necessary packages/functions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from nltk.corpus import stopwords
import string
from unidecode import unidecode
import random
import itertools
import csv


# %%
PATH = "../data/"

#### Reading in the data.
content = pd.read_csv(f"{PATH}content_filtered.csv")
correlations = pd.read_csv(f"{PATH}correlations.csv")
#sample_submission = pd.read_csv("/kaggle/input/learning-equality-curriculum-recommendations/sample_submission.csv")
topics = pd.read_csv(f"{PATH}topics_filtered.csv")



# %%
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

# %%
#### Apply combine() to our data
corr_topics, corr_content, index_to_topic, index_to_content = combine(correlations, topics, content)

# %%
#### Create a stopword removal function to remove stopwords for each language
import nltk
nltk.download('stopwords')
# Dictionary of languages found in our data
lang_dict = {
    "en":"english",
    "es":"spanish",
    "it":"italian",
    'pt':"portuguese",
    'mr':'marathi',
    'bg':'bulgarian',
    'gu':'gujarati',
    'sw':'swahili',
    'hi':'hindi',
    'ar':'arabic',
    'bn':'bengali',
    'as':'assamese',
    'zh':'chinese',
    'fr':'french',
    'km':'khmer',
    'pl':'polish',
    'ta':'tamil',
    'or':'oriya',
    'ru':'russian',
    'kn':'kannada',
    'swa':'swahili',
    'my':'burmese',
    'pnb':'punjabi',
    'fil':'filipino',
    'tr':'turkish',
    'te':'telugu',
    'ur':'urdu',
    'fi':'finnish',
    'pn':'unknown',
    'mu':'unknown'}

# List of languages supported by the natural language tool kit (NLTK) module.
supported_languages = stopwords.fileids()

def remove_stopwords(text):
    '''
    Checks language of text then removes stopwords from that language if supported.
    '''
    lang_code = text[0:2]
    if lang_dict[lang_code] in supported_languages:
        for word in stopwords.words(lang_dict[lang_code]):
            text = text.replace(' ' + word + ' ', ' ')
    return text

# %%
#### Apply remove_stopwords() to our data
corr_topics["features"] = corr_topics.features.apply(remove_stopwords)
corr_content["features"] = corr_content.features.apply(remove_stopwords)

# %%
#### Create train/test indices for our data 
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

# %%
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

# %%
#### Create labels
train_labels = np.array((train_topics.id == train_content.id).astype(int))
test_labels = np.array((test_topics.id == test_content.id).astype(int))

# %%
#### Convert data to tensors
train_topics = tf.data.Dataset.from_tensor_slices(tf.cast(train_topics.features, tf.string))
train_content = tf.data.Dataset.from_tensor_slices(tf.cast(train_content.features, tf.string))
train_labels = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int32))

test_topics = tf.data.Dataset.from_tensor_slices(tf.cast(test_topics.features, tf.string))
test_content = tf.data.Dataset.from_tensor_slices(tf.cast(test_content.features, tf.string))
test_labels = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int32))

# %%
#### Combine data into TensorFlow Datasets
#### Perfectly shuffle, batch, cache, and prefetch our new datasets
train_ds = tf.data.Dataset.zip(
    ((train_topics, train_content), train_labels)
)

train_ds = train_ds.shuffle(buffer_size = train_ds.cardinality().numpy()).batch(batch_size = 64).cache().prefetch(tf.data.experimental.AUTOTUNE)

test_ds = tf.data.Dataset.zip(
    ((test_topics, test_content), test_labels)
)

test_ds = test_ds.shuffle(buffer_size = test_ds.cardinality().numpy()).batch(batch_size = 64).cache().prefetch(tf.data.experimental.AUTOTUNE)

# %%
#### Create Text Vectorization Layer
# Hyperparameters
VOCAB_SIZE = 1000000
MAX_LEN = 50

def my_standardize(text): 
    '''
    A text standardization function that is applied for every element in the vectorize layer. 
    '''
    text = tf.strings.lower(text, encoding='utf-8') #lowercase
    text = tf.strings.regex_replace(text, f"([{string.punctuation}])", r" ") #remove punctuation
    text = tf.strings.regex_replace(text, '\n', "") #remove newlines
    text = tf.strings.regex_replace(text, ' +', " ") #remove 2+ whitespaces
    text = tf.strings.strip(text) #remove leading and tailing whitespaces
    return text

vectorize_layer = TextVectorization(
    standardize = my_standardize,
    split = "whitespace",
    max_tokens = VOCAB_SIZE + 2,
    output_mode = 'int',
    output_sequence_length = MAX_LEN
)

# %%
#### Adapt text vectorization layer to our data
vectorize_layer.adapt(pd.concat([corr_topics["features"], corr_content["features"]]))

# %%
inp_topics = Input((1, ), dtype=tf.string)
inp_content = Input((1, ), dtype=tf.string)

vectorized_topics = vectorize_layer(inp_topics)
vectorized_content = vectorize_layer(inp_content)

snn = Sequential([ 
  Embedding(VOCAB_SIZE, 256),
  GlobalAveragePooling1D(),
  Flatten(),
  Dense(128, activation='relu'),
])

snn_content = snn(vectorized_content)
snn_topics = snn(vectorized_topics)

concat = Concatenate()([snn_topics, snn_content])

dense = Dense(64, activation='relu')(concat)

output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[inp_topics, inp_content], outputs=output)

model.summary()

# %%
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=tf.keras.metrics.AUC())

# %%
model.fit(train_ds, epochs=5, verbose=1)

# %%
model.evaluate(test_ds, verbose=1)

# %%
#### Antijoin topics with correlations data, since we don't have to predict those topics
outer_joined = topics.merge(correlations, how='outer', left_on='id', right_on='topic_id', indicator=True)
topics = outer_joined[(outer_joined._merge == 'left_only')].drop('_merge', axis=1)

#### Fill missing values and concatenate text to features column 
topics = topics.fillna("")
topics_ids = topics.id.values
topics_lang = topics.language
topics_index = topics.index
topics_features = topics["language"] + ' ' + topics["channel"] + ' ' + topics["title"] + ' ' + topics["description"]
del topics

#### Repeat for content, except we keep all content data
content = content.fillna("")
content_ids = content.id.values
content_index = content.index
content_lang = content.language
content_features = content["language"] + ' ' + content["title"] + ' ' + content["description"] + ' ' + content["text"]
del content

index_to_content = pd.Series(content_ids, index=content_index).to_dict()
index_to_topic = pd.Series(topics_ids, index=topics_index).to_dict()

# %%
#### Remove stopwords
topics_features = topics_features.apply(remove_stopwords)
content_features = content_features.apply(remove_stopwords)

# %%
#### Write predictions to output_file
THRESHOLD = 0.99994

output_file = "submission.csv"
f = open(output_file, 'w')

writer = csv.writer(f)
writer.writerow(["topic_id", "content_ids"])

for i in topics_features.index:
    temp_content = tf.data.Dataset.from_tensor_slices(
        tf.cast(content_features[content_lang == topics_lang[i]], tf.string)
    )
    temp_topic = tf.data.Dataset.from_tensor_slices(
        tf.cast(np.repeat(topics_features[i], len(temp_content)), tf.string)
    )
    temp_ds = tf.data.Dataset.zip(((temp_topic, temp_content), ))\
        .batch(batch_size=64)\
            .cache()\
                .prefetch(tf.data.experimental.AUTOTUNE)
    matches = model.predict(temp_ds, verbose=0)
    matches = [i for i in range(len(matches)) if matches[i] > THRESHOLD]
    matches = " ".join([index_to_content[x] for x in matches])
    writer.writerow([index_to_topic[i], matches])

#### Add given correlations data
writer.writerows([correlations.topic_id, correlations.content_ids])    

f.close()


