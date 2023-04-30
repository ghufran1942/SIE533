import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Read the CSV files
topics_df = pd.read_csv("data/topics.csv")
correlations_df = pd.read_csv("data/correlations.csv")

# Extract relevant information
train_texts = topics_df["content"].tolist()
train_labels = topics_df["topic"].tolist()
val_texts = correlations_df["content"].tolist()
val_labels = correlations_df["topic"].tolist()

# Tokenization and encoding
def encode_data(texts, labels, tokenizer, max_seq_length, device):
    input_ids, attention_masks, target_labels = [], [], []

    for text, label in zip(texts, labels):
        encoded = tokenizer.encode_plus(
            text,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        target_labels.append(label)

    return {
        "input_ids": torch.cat(input_ids, dim=0).to(device),
        "attention_masks": torch.cat(attention_masks, dim=0).to(device),
        "labels": torch.tensor(target_labels).to(device),
    }

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_seq_length = 128
encoded_train_data = encode_data(train_texts, train_labels, tokenizer, max_seq_length, device)
encoded_val_data = encode_data(val_texts, val_labels, tokenizer, max_seq_length, device)

# Model creation
num_labels = len(set(train_labels))
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model.to(device)

# Create data loaders
train_loader, val_loader = create_data_loaders(encoded_train_data, encoded_val_data, batch_size=16)

# Fine-tuning
epochs = 3
learning_rate = 2e-5
warmup_steps = 0
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader) * epochs)

train_model(model, optimizer, scheduler, train_loader, val_loader, epochs)

# Prediction function
def predict_topic(text, model, tokenizer, max_seq_length, device):
    model.eval()

    encoded_text = tokenizer.encode_plus(
        text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    with torch.no_grad():
        input_ids = encoded_text["input_ids"].to(device)
        attention_mask = encoded_text["attention_mask"].to(device)
        output = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(output.logits, dim=1)

    predicted_topic = torch.argmax(probabilities, dim=1).item()

    return predicted_topic

input_text = "Sample text to predict the topic"
predicted_topic = predict_topic(input_text, model, tokenizer, max_seq_length, device)
print(f"Predicted topic: {predicted_topic}")
