"""
model_training.py

This module implements a BERT-based classification model for detecting abusive content
in sports commentary. It includes dataset handling, model training, and evaluation
functionality.

The model achieved the following performance metrics:
- Training accuracy: 90.73%
- Validation accuracy: 89.93%
- Test accuracy: 89.60%
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


df = pd.read_csv('synthetic_tweets_20241203_000313.csv')
texts = df['text'].values
labels = df['label'].values

print(f"Loaded {len(texts)} samples")
print(f"Distribution of labels: \n{pd.Series(labels).value_counts()}")

class AbuseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': predictions,
        'true_labels': true_labels
    }

def train_model(train_loader, val_loader, model, device, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_val_accuracy = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = len(train_loader)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

        train_metrics = compute_metrics(model, train_loader, device)
        val_metrics = compute_metrics(model, val_loader, device)

        print(f"Epoch {epoch+1} Summary:")
        print(f"Training Loss: {train_metrics['loss']:.4f}")
        print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")

        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()

    return best_model_state

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Create datasets
train_dataset = AbuseDataset(X_train, y_train, tokenizer)
val_dataset = AbuseDataset(X_val, y_val, tokenizer)
test_dataset = AbuseDataset(X_test, y_test, tokenizer)

# Create data loaders with larger batch size for efficiency
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Set device and move model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Train the model
print("Starting training...")
best_model_state = train_model(train_loader, val_loader, model, device)

# Load best model for final evaluation
model.load_state_dict(best_model_state)

# Evaluate on test set
test_metrics = compute_metrics(model, test_loader, device)
print("\nFinal Test Results:")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test Loss: {test_metrics['loss']:.4f}")

def predict_abuse(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probabilities[0].cpu().numpy()

# Test some examples
print("\nTesting model on new examples...")
test_texts = [
    "What a fantastic game! The team played brilliantly today!",
    "You're absolutely worthless, get out of our club!",
    "Great effort by everyone, proud of the team!",
    "The referee needs to get their eyes checked, completely blind!"
]

for text in test_texts:
    probs = predict_abuse(text)
    print(f"\nText: {text}")
    print(f"Probability of non-abusive: {probs[0]:.4f}")
    print(f"Probability of abusive: {probs[1]:.4f}")
