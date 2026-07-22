import torch
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import config
from model.baseline import Baseline
from model.ruBERT import ruBERT
from transformers import BertTokenizerFast
import torch.nn.functional as F

MODELS = {}

def load_models():
    global MODELS
    if MODELS:
        return MODELS

    # 1. MLP (Baseline)
    try:
        from utils.tf_idf import vectorize_texts
        vectorizer = vectorize_texts(["load"], ["load"], save=False)[2]

        model_mlp = Baseline(
            input_dim=5000,
            num_classes=11,
            hidden_dim=256
        )
        model_mlp.load_state_dict(torch.load(config.MLP_MODEL_PATH, map_location=config.DEVICE))
        model_mlp.eval()
        MODELS["mlp"] = (model_mlp, vectorizer)
        print("MLP loaded")
    except Exception as e:
        print(f"MLP error: {e}")

    # 2. ruBERT
    try:
        model_bert = ruBERT(num_classes=11)
        model_bert.load_state_dict(torch.load(config.BERT_MODEL_PATH, map_location=config.DEVICE))
        model_bert.eval()

        tokenizer_bert = BertTokenizerFast.from_pretrained(config.BERT_TOKENIZER_PATH)
        MODELS["bert"] = (model_bert, tokenizer_bert)
        print("ruBERT loaded")
    except Exception as e:
        print(f"ruBERT error: {e}")

    # 3. ruBERT Tiny
    try:
        model_bert_tiny = ruBERT(model_name=config.BERT_TINY_MODEL_NAME, num_classes=11)
        model_bert_tiny.load_state_dict(torch.load(config.BERT_TINY_MODEL_PATH, map_location=config.DEVICE))
        model_bert_tiny.eval()

        tokenizer_bert_tiny = BertTokenizerFast.from_pretrained(config.BERT_TINY_TOKENIZER_PATH)
        MODELS["bert_tiny"] = (model_bert_tiny, tokenizer_bert_tiny)
        print("ruBERT Tiny loaded")
    except Exception as e:
        print(f"ruBERT Tiny error: {e}")

    return MODELS


def classify_text(text: str, model_type: str):
    if not MODELS:
        load_models()

    model, tokenizer = MODELS.get(model_type, (None, None))
    if model is None:
        return {"error": f"Модель '{model_type}' не найдена."}

    if model_type == "mlp":
        # MLP requires vectorization
        from utils.tf_idf import vectorize_texts
        vectorizer = tokenizer
        X = vectorizer.transform([text]).toarray()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(X_t)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

    else:
        # BERT / BERT Tiny
        encoding = tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

    class_names = [
        "climate", "conflicts", "culture", "economy", "gloss",
        "health", "politics", "science", "society", "sports", "travel"
    ]

    return {
        "pred_class": pred_class,
        "confidence": confidence * 100,
        "class_name": class_names[pred_class],
        "proba": probs[0].tolist(),
        "class_names_probs": [(name, prob * 100) for name, prob in zip(class_names, probs[0].tolist())]
    }