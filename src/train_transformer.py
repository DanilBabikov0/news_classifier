import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

import config
from model.ruBERT import ruBERT
from utils.create_dataloader import get_transformer_loaders
from utils.metrics import compute_metrics, plot_confusion_matrix


def train():
    print("Обучение ruBERT...")
    train_loader, test_loader, tokenizer = get_transformer_loaders(
        model_name=config.BERT_MODEL_NAME,
        batch_size=config.BERT_BATCH_SIZE,
        max_length=config.BERT_MAX_LENGTH
    )

    model = ruBERT(
        model_name=config.BERT_MODEL_NAME,
        num_classes=len(config.CLASS_NAMES),
        dropout=0.3
    ).to(config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=config.BERT_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    num_epochs = config.BERT_EPOCHS

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} — Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), config.BASELINE_MODEL / "transformer_model.pth")
    print("Модель сохранена.")


if __name__ == "__main__":
    train()