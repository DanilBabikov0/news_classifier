import os
import config

def check_data_exists():
    missing = []

    if not os.path.exists(config.TRAIN_CSV):
        missing.append(f"train CSV: {config.TRAIN_CSV}")
    if not os.path.exists(config.TEST_CSV):
        missing.append(f"test CSV: {config.TEST_CSV}")

    if missing:
        print("Отсутствуют данные:")
        for f in missing:
            print(f"   - {f}")
        print("Запустите: python3 src.data_loader\n")
        return False
    return True