import torch
import os
import json
import numpy as np

import config
from model.ruBERT import ruBERT
from transformers import BertTokenizerFast

def load_bert_model():
    # 1. Проверка существования модели
    model_path = config.BERT_MODEL_PATH
    tokenizer_path = config.BERT_TOKENIZER_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}.")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Токенизатор не найден: {tokenizer_path}.")

    # 2. Загрузка токенизатора
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    # 3. Инициализация модели
    model = ruBERT(
        model_name=config.BERT_MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout=0.3
    ).to(config.DEVICE)

    # 4. Загрузка весов
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    print(f"Модель ruBERT загружена: {model_path}")
    print(f"Токенизатор загружен: {tokenizer_path}")
    print(f"Используется устройство: {config.DEVICE}")

    return model, tokenizer


def predict_bert(text: str, model, tokenizer):
    # 1. Токенизация
    encoding = tokenizer(
        text,
        max_length=config.BERT_MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 2. Перемещение на устройство
    input_ids = encoding["input_ids"].to(config.DEVICE)
    attention_mask = encoding["attention_mask"].to(config.DEVICE)

    # 3. Инференс
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        proba = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(proba, dim=1).item()
        confidence = proba[0][pred_class].item()

    return pred_class, confidence, proba.cpu().numpy()[0]


def print_predictions(pred_class: int, confidence: float, proba: np.ndarray):
    class_names = config.CLASS_NAMES
    print("\n" + "=" * 50)
    print("Результаты классификации:")
    print("=" * 50)
    print(f"Предсказанный класс: {pred_class} → {class_names[pred_class]}")
    print(f"Уверенность: {confidence:.2%}")
    print("\nВероятности по классам (порог > 1%):")
    for i, p in enumerate(proba):
        if p > 0.01:
            print(f"  {i:2d} ({class_names[i]:12s}): {p:.2%}")


def main():
    print("=" * 60)
    print("Инференс ruBERT для классификации новостей")
    print("=" * 60)

    try:
        model, tokenizer = load_bert_model()
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        print("Запустите: python -m src.train_transformer")
        return

    class_names = config.CLASS_NAMES
    print(f"\nДоступно классов: {len(class_names)} → {class_names}\n")

    print("Введите текст для классификации.")
    print("   'quit', 'exit', 'q' — выход.\n")

    while True:
        try:
            text = input("Введите текст: ").strip()
            if not text:
                print("Пустой текст, повторите.")
                continue

            if text.lower() in ['quit', 'exit', 'q']:
                print("До встречи")
                break

            pred_class, confidence, proba = predict_bert(text, model, tokenizer)
            print_predictions(pred_class, confidence, proba)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nДо встречи")
            break
        except Exception as e:
            print(f"Ошибка при обработке: {e}")


if __name__ == "__main__":
    main()