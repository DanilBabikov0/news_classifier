import torch
import numpy as np
import os
import config

from utils.metrics import (
    compute_all_metrics,
    save_metrics,
    plot_confusion_matrix,
    plot_comparison_summary
)

from model.baseline import Baseline
from model.ruBERT import ruBERT
from transformers import BertTokenizerFast


def evaluate_model(model, test_loader, model_name):
    print(f"\nОценка модели: {model_name}")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            if model_name == "baseline":
                X_batch, y_batch = batch
                X_batch = X_batch.to(config.DEVICE)
                y_batch = y_batch.to(config.DEVICE)
                outputs = model(X_batch)
                labels = y_batch
            else:
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                labels = batch["labels"].to(config.DEVICE)
                outputs = model(input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    model_plot_dir = os.path.join(config.PLOT_DIR, model_name)
    os.makedirs(model_plot_dir, exist_ok=True)

    # Confusion Matrix
    cm_path = os.path.join(model_plot_dir, "cm.png")
    plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES, save_path=cm_path, title=f"{model_name}")

    # Метрики в JSON
    metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names=config.CLASS_NAMES)
    save_metrics(metrics, model_name, model_plot_dir)

    # Вывод
    print(f"{model_name}:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Macro F1: {metrics['macro_f1']:.4f}")
    print(f"   ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")

    return metrics


def main():
    print("=" * 70)
    print("ОЦЕНКА И СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)

    # === 1. MLP ===
    from utils.create_dataloader import load_processed_data, create_dataloaders
    from utils.tf_idf import vectorize_texts

    print("\nЗагрузка MLP данных...")
    X_train, y_train, X_test, y_test = load_processed_data()
    X_train_vec, X_test_vec, _ = vectorize_texts(X_train, X_test, save=True)
    _, test_loader_mlp = create_dataloaders(X_train_vec, y_train, X_test_vec, y_test)

    model_mlp = Baseline(
        input_dim=config.MAX_FEATURES,
        num_classes=config.NUM_CLASSES,
        hidden_dim=256
    ).to(config.DEVICE)
    model_mlp.load_state_dict(torch.load(config.MLP_MODEL_PATH, map_location=config.DEVICE))
    metrics_mlp = evaluate_model(model_mlp, test_loader_mlp, "baseline")

    # === 2. ruBERT ===
    from utils.create_dataloader import get_transformer_loaders

    print("\nЗагрузка ruBERT данных...")
    test_loader_rubert, _, tokenizer = get_transformer_loaders(
        model_name=config.BERT_MODEL_NAME,
        batch_size=config.BERT_BATCH_SIZE,
        max_length=config.BERT_MAX_LENGTH
    )

    model_rubert = ruBERT(
        model_name=config.BERT_MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout=0.3
    ).to(config.DEVICE)
    model_rubert.load_state_dict(torch.load(config.BERT_MODEL_PATH, map_location=config.DEVICE))
    metrics_rubert = evaluate_model(model_rubert, test_loader_rubert, "rubert")

    # === 3. ruBERT-tiny ===
    print("\nЗагрузка ruBERT-tiny данных...")
    test_loader_rubert_tiny, _, _ = get_transformer_loaders(
        model_name=config.BERT_TINY_MODEL_NAME,
        batch_size=config.BERT_BATCH_SIZE,
        max_length=config.BERT_MAX_LENGTH
    )

    model_rubert_tiny = ruBERT(
        model_name=config.BERT_TINY_MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout=0.3
    ).to(config.DEVICE)
    model_rubert_tiny.load_state_dict(torch.load(config.BERT_TINY_MODEL_PATH, map_location=config.DEVICE))
    metrics_rubert_tiny = evaluate_model(model_rubert_tiny, test_loader_rubert_tiny, "rubert_tiny")

    # === 4. Сравнение моделей (только графики и таблица) ===
    metrics_list = [metrics_mlp, metrics_rubert, metrics_rubert_tiny]
    models = ["baseline", "rubert", "rubert_tiny"]

    print("\n" + "=" * 70)
    print("ГЕНЕРАЦИЯ СРАВНИТЕЛЬНЫХ ГРАФИКОВ")
    print("=" * 70)

    plot_comparison_summary(metrics_list, models, config.CLASS_NAMES, config.PLOT_DIR + "/models_comparison")

    print("\nГотово!")
    print("   → Confusion Matrices и метрики сохранены в plots/{model}/")
    print("   → Сравнение — plots/models_comparison/")


if __name__ == "__main__":
    main()