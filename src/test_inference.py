import torch
import pandas as pd
from joblib import load
import os

import config
from models import Baseline
from utils.tf_idf import vectorize_texts

def load_model_and_vectorizer():
    model = Baseline(
        input_dim=config.MAX_FEATURES,
        num_classes=config.NUM_CLASSES,
        hidden_dim=256
    ).to(config.DEVICE)
    
    model_path = os.path.join(config.BASELINE_MODEL, "baseline_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}. Обучите модель через run_baseline.py")
    
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print(f"Модель загружена: {model_path}")

    vectorizer_path = os.path.join(config.FEATURES_DIR, "tfidf_vectorizer.joblib")
    vectorizer = load(vectorizer_path)
    print(f"Векторайзер загружен: {vectorizer_path}")
    
    return model, vectorizer


def predict_text(text: str, model, vectorizer):
    vec = vectorizer.transform([text]).toarray()
    vec_t = torch.tensor(vec, dtype=torch.float32).to(config.DEVICE)

    with torch.no_grad():
        output = model(vec_t)
        proba = torch.softmax(output, dim=1)
        pred_class = torch.argmax(proba, dim=1).item()
        confidence = proba[0][pred_class].item()

    return pred_class, confidence, proba.cpu().numpy()[0]


def get_class_names():
    return config.CLASS_NAMES

def main():
    print("=" * 60)
    print("Тестирование модели: инференс на новых данных")
    print("=" * 60)

    try:
        model, vectorizer = load_model_and_vectorizer()
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    class_names = get_class_names()
    
    print("\nВы можете вводить тексты для классификации.")
    print("   Введите 'quit', 'exit' или 'q' для выхода.\n")

    while True:
        try:
            text = input("Введите текст: ").strip()
            if not text:
                print("Пустой текст, повторите.")
                continue

            if text.lower() in ['quit', 'exit', 'q']:
                print("До встречи!")
                break

            # Предсказание
            pred_class, confidence, proba = predict_text(text, model, vectorizer)
            
            print(f"\nРезультат:")
            print(f"   Класс: {pred_class} → {class_names[pred_class]}")
            print(f"   Уверенность: {confidence:.2%}")
            print(f"   Вероятности по классам:")
            for i, p in enumerate(proba):
                if p > 0.01:
                    print(f"      {i} ({class_names[i]}): {p:.2%}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nДо встречи")
            break
        except Exception as e:
            print(f"Ошибка при обработке: {e}")


if __name__ == "__main__":
    main()