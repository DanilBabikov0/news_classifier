import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import json
import os

import config
from model.ruBERT import ruBERT
from utils.create_dataloader import get_transformer_loaders
from utils.metrics import plot_training_curves


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    return avg_loss, all_preds, all_labels


def main():
    # 1. Загрузка данных
    print("Загрузка токенизатора и данных...")
    train_loader, test_loader, tokenizer = get_transformer_loaders(
        model_name=config.BERT_MODEL_NAME,
        batch_size=config.BERT_BATCH_SIZE,
        max_length=config.BERT_MAX_LENGTH
    )

    # 2. Модель
    print(f"Инициализация ruBERT на {config.DEVICE}...")
    model = ruBERT(
        model_name=config.BERT_MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout=0.3
    ).to(config.DEVICE)

    # 3. Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.BERT_LEARNING_RATE)

    # 4. История обучения
    history = {
        "train_loss": [],
        "val_loss": []
    }

    # 5. Обучение
    print("\nНачало обучения...")
    num_epochs = config.BERT_EPOCHS
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_preds, val_labels = evaluate(model, test_loader, criterion, config.DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"Улучшение! Сохранена лучшая модель. Val Loss: {val_loss:.4f}")

    # 6. Загрузка лучшей модели
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Лучшая Val Loss: {best_val_loss:.4f}")

    # 7. Сохранение модели
    os.makedirs(config.BERT_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.BERT_MODEL_DIR, "bert_model.pth")
    tokenizer_path = os.path.join(config.BERT_MODEL_DIR, "tokenizer")

    torch.save(model.state_dict(), model_path)
    model.bert.save_pretrained(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)

    print(f"\nМодель сохранена: {model_path}")
    print(f"Токенизатор сохранен: {tokenizer_path}")

    # 8. Сохранение истории обучения
    os.makedirs(os.path.join(config.PLOT_DIR, "rubert"), exist_ok=True)
    history_path = os.path.join(config.PLOT_DIR, "rubert", "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"История обучения сохранена: {history_path}")

    # 9. График обучения
    plot_path = os.path.join(config.PLOT_DIR, "rubert", "training_curves.png")
    plot_training_curves(history, save_path=plot_path)

    print("\nОбучение завершено! Сохранено:")
    print(f"   - Модель: {config.BERT_MODEL_DIR}/bert_model.pth")
    print(f"   - График обучения: {config.PLOT_DIR}/rubert/training_curves.png")


if __name__ == "__main__":
    main()