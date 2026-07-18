import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from models import Baseline
from utils.tf_idf import vectorize_texts
from utils.create_dataloader import load_processed_data, create_dataloaders
import os

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Validation", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    # 1. Загрузка данных
    print("Загрузка данных...")
    X_train, y_train, X_test, y_test = load_processed_data()

    # 2. Векторизация
    print("Векторизация (TF-IDF)...")
    X_train_vec, X_test_vec, vectorizer = vectorize_texts(X_train, X_test, save=True)

    # 3. DataLoader
    print("Создание DataLoader...")
    train_loader, test_loader = create_dataloaders(X_train_vec, y_train, X_test_vec, y_test)

    # 4. Модель
    print(f"Инициализация модели на {config.DEVICE}...")
    model = Baseline(
        input_dim=config.MAX_FEATURES,
        num_classes=config.NUM_CLASSES,
        hidden_dim=256  # можно сделать 512, если нужна большая модель
    ).to(config.DEVICE)

    # 5. Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 6. Обучение
    print("Начало обучения...")
    best_val_acc = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = evaluate(model, test_loader, criterion, config.DEVICE)

        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, "baseline_model.pth"))
            print(f"Модель сохранена! Best Val Acc: {best_val_acc:.2f}%")

    print("Обучение завершено")


if __name__ == "__main__":
    main()