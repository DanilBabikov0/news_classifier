import os
import torch

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_CLASSES = 11

EARLY_STOPPING_PATIENCE = 5

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
BASELINE_MODEL = os.path.join(MODELS_DIR, "baseline")
FEATURES_DIR = os.path.join(MODELS_DIR, "features")

PLOT_DIR = os.path.join(BASE_DIR, "plots")

TRAIN_CSV = os.path.join(DATA_PROCESSED, "rus_news_classifier_train.csv")
TEST_CSV = os.path.join(DATA_PROCESSED, "rus_news_classifier_test.csv")