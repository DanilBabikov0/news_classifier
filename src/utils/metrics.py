import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import os

def compute_metrics(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    return {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": report
    }

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion Matrix saved to {save_path}")
    plt.show()

def plot_training_curves(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="s")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training Curves saved to {save_path}")
    plt.show()