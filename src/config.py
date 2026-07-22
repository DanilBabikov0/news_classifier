import os
import torch

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# =========== tf idf ===========
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# =========== MLP ===========
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# =========== BERT ===========
BERT_MODEL_NAME = "ai-forever/ruBERT-base"
BERT_TINY_MODEL_NAME = "cointegrated/rubert-tiny2"
BERT_MAX_LENGTH = 64
BERT_BATCH_SIZE = 8
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 2e-5
BERT_WEIGHT_DECAY = 0.01

# =========== ДАТАСЕТ ===========
DATASET_NAME = "data-silence/rus_news_classifier"
TRAIN_CSV_NAME = "rus_news_classifier_train.csv"
TEST_CSV_NAME = "rus_news_classifier_test.csv"

X_NAME = "news"
Y_NAME = "labels"
CLASS_NAMES = [
        "climate",
        "conflicts",
        "culture",
        "economy",
        "gloss",
        "health",
        "politics",
        "science",
        "society",
        "sports",
        "travel"
    ]
NUM_CLASSES = len(CLASS_NAMES)

# =========== ПАПКИ ===========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
BASELINE_MODEL_DIR = os.path.join(MODELS_DIR, "baseline")
BERT_MODEL_DIR = os.path.join(MODELS_DIR, "bert")
BERT_TINY_MODEL_DIR = os.path.join(MODELS_DIR, "bert_tiny")

MLP_MODEL_PATH = os.path.join(BASELINE_MODEL_DIR, "baseline_model.pth")
MLP_VECTORIZER_PATH = os.path.join(BASELINE_MODEL_DIR, "vectorizer")

BERT_MODEL_PATH = os.path.join(BERT_MODEL_DIR, "bert_model.pth")
BERT_TINY_MODEL_PATH = os.path.join(BERT_TINY_MODEL_DIR, "bert_model.pth")
BERT_TOKENIZER_PATH = os.path.join(BERT_MODEL_DIR, "tokenizer")
BERT_TINY_TOKENIZER_PATH = os.path.join(BERT_TINY_MODEL_DIR, "tokenizer")

PLOT_DIR = os.path.join(BASE_DIR, "plots")

TRAIN_CSV = os.path.join(DATA_PROCESSED, TRAIN_CSV_NAME)
TEST_CSV = os.path.join(DATA_PROCESSED, TEST_CSV_NAME)