import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    f1_score
)

def compute_all_metrics(y_true, y_pred, y_proba=None, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": report
    }

    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    if y_proba is not None:
        try:
            metrics["roc_auc_macro"] = roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
            metrics["roc_auc_weighted"] = roc_auc_score(y_true, y_proba, multi_class="ovo", average="weighted")
        except Exception as e:
            print(f"[!] ROC AUC failed: {e}")
            metrics["roc_auc_macro"] = metrics["roc_auc_weighted"] = np.nan
    else:
        metrics["roc_auc_macro"] = metrics["roc_auc_weighted"] = np.nan

    return metrics


def save_metrics(metrics, model_name, output_dir):
    path = os.path.join(output_dir, f"metrics_{model_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Метрики сохранены: {path}")


def load_metrics(model_name, metrics_dir):
    path = os.path.join(metrics_dir, f"metrics_{model_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[!] Метрики {model_name} не найдены: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title or "Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion Matrix saved: {save_path}")
    plt.show()


def plot_comparison_bar(models, values, title, ylabel, filename):
    x = np.arange(len(models))
    width = 0.6

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, values, width, color=['#4C72B0', '#55A868', '#C44E58'])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def plot_per_class_f1_comparison(metrics_list, models, class_names, save_path=None):
    df = []
    for model_name, metrics in zip(models, metrics_list):
        per_class = metrics["per_class"]
        f1_scores = [per_class[class_name]["f1-score"] for class_name in class_names]
        df.extend([{"Model": model_name, "Class": class_name, "F1": f1} for class_name, f1 in zip(class_names, f1_scores)])

    df = pd.DataFrame(df)

    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x="Class", y="F1", hue="Model")
    plt.title("Сравнение F1-score по классам")
    plt.xlabel("Класс")
    plt.ylabel("F1-score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Per-class F1 saved: {save_path}")
    plt.show()


def plot_comparison_summary(metrics_list, models, class_names, output_dir):
    # 1. Основные метрики
    accuracy = [m["accuracy"] for m in metrics_list]
    macro_f1 = [m["macro_f1"] for m in metrics_list]
    balanced_acc = [m["balanced_accuracy"] for m in metrics_list]
    cohen_kappa = [m["cohen_kappa"] for m in metrics_list]
    roc_auc = [m["roc_auc_macro"] for m in metrics_list]

    summary = pd.DataFrame({
        "Model": models,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Balanced Acc": balanced_acc,
        "Cohen's Kappa": cohen_kappa,
        "ROC-AUC (macro)": roc_auc
    })

    summary_path = os.path.join(output_dir, "summary_comparison.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Сводная таблица сохранена: {summary_path}")

    # 2. Графики
    plot_comparison_bar(models, accuracy, "Accuracy", "Score", os.path.join(output_dir, "compare_accuracy.png"))
    plot_comparison_bar(models, macro_f1, "Macro F1", "Score", os.path.join(output_dir, "compare_macro_f1.png"))
    plot_comparison_bar(models, balanced_acc, "Balanced Accuracy", "Score", os.path.join(output_dir, "compare_balanced_acc.png"))
    plot_comparison_bar(models, cohen_kappa, "Cohen’s Kappa", "Score", os.path.join(output_dir, "compare_cohen_kappa.png"))

    # 3. Per-class F1
    plot_per_class_f1_comparison(metrics_list, models, class_names, os.path.join(output_dir, "compare_per_class_f1.png"))

    print("\nСравнение завершено!")
    print(f"   → {summary_path}")
    print(f"   → Графики сохранены в: {output_dir}")

def plot_training_curves(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", color="#4C72B0")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s", color="#55A868")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o", color="#4C72B0")
    ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="s", color="#55A868")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved: {save_path}")

    plt.show()