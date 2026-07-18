import os
import torch

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

X_NAME = "news"
Y_NAME = "labels"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "baseline")

TRAIN_CSV = os.path.join(DATA_PROCESSED, "rus_news_classifier_train.csv")
TEST_CSV = os.path.join(DATA_PROCESSED, "rus_news_classifier_test.csv")