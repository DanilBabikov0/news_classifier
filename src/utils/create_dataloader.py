import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizerFast

import config

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def get_transformer_loaders(model_name=config.BERT_MODEL_NAME, batch_size=16, max_length=128):
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    X_train = train_df[config.X_NAME].values
    y_train = train_df[config.Y_NAME].values
    X_test = test_df[config.X_NAME].values
    y_test = test_df[config.Y_NAME].values

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    train_dataset = NewsDataset(X_train, y_train, tokenizer, max_length)
    test_dataset = NewsDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer


def load_processed_data():
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    X_train = train_df[config.X_NAME].values
    y_train = train_df[config.Y_NAME].values
    X_test = test_df[config.X_NAME].values
    y_test = test_df[config.Y_NAME].values
    return X_train, y_train, X_test, y_test



def create_dataloaders(X_train, y_train, X_test, y_test):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, test_loader