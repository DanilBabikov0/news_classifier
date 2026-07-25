import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os

import config
from model.baseline import Baseline
from utils.tf_idf import vectorize_texts
from utils.create_dataloader import load_processed_data, create_dataloaders
from utils.metrics import plot_training_curves
from utils.check import check_data_exists

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
    if check_data_exists() == False:
        return 0
    
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
        hidden_dim=256
    ).to(config.DEVICE)

    # 5. Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 6. Обучение
    print("Начало обучения...")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    patience = config.EARLY_STOPPING_PATIENCE
    best_val_acc = 0
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = evaluate(model, test_loader, criterion, config.DEVICE)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
            print(f"Улучшение Best Val Acc: {best_val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")
        if epochs_without_improvement >= patience:
            print(f"Early stopping Лучшая Val Acc: {best_val_acc:.2f}%")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), os.path.join(config.BASELINE_MODEL_DIR, "baseline_model.pth"))
        print(f"Модель сохранена. Best Val Acc: {best_val_acc:.2f}%")

    print("Обучение завершено")

    os.makedirs(os.path.join(config.PLOT_DIR, "baseline"), exist_ok=True)

    history_path = os.path.join(config.PLOT_DIR, "baseline", "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"История обучения сохранена: {history_path}")

    plot_path = os.path.join(config.PLOT_DIR, "baseline", "training_curves.png")
    plot_training_curves(history, save_path=plot_path)

    print("\nОбучение завершено! Сохранено:")
    print(f"   - Модель: {config.BASELINE_MODEL_DIR}/baseline_model.pth")
    print(f"   - График обучения: {config.PLOT_DIR}/baseline/training_curves.png")

if __name__ == "__main__":
    main()