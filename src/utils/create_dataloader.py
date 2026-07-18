import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config

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