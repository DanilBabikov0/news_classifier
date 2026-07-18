from datasets import load_dataset
import pandas as pd
import os
import utils.preprocessing as preprocessing
import config


def load_and_save_raw_datasets():
    dataset = load_dataset("data-silence/rus_news_classifier")

    dataset_train = pd.DataFrame(dataset['train'])
    dataset_test = pd.DataFrame(dataset['test'])

    os.makedirs(config.DATA_RAW, exist_ok=True)

    train_path = os.path.join(config.DATA_RAW, "rus_news_classifier_train.csv")
    test_path = os.path.join(config.DATA_RAW, "rus_news_classifier_test.csv")

    dataset_train.to_csv(train_path, index=False)
    dataset_test.to_csv(test_path, index=False)

    return train_path, test_path


def process_datasets(train_path: str, test_path: str):
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)

    preprocessing.load_and_clean(
        data_path=train_path,
        save_path=config.TRAIN_CSV
    )

    preprocessing.load_and_clean(
        data_path=test_path,
        save_path=config.TEST_CSV
    )


def load_data():
    train_path, test_path = load_and_save_raw_datasets()
    process_datasets(train_path, test_path)


if __name__ == "__main__":
    load_data()