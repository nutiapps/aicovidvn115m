import gdown
from config import Config
import numpy as np
import zipfile
import shutil

if __name__ == '__main__':
    Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

    # download train dataset
    gdown.download(
        Config.TRAIN_DATASET_FILE_PATH,
        str(Config.DATASET_PATH / "aicv115m_public_train.zip"),
    )

    # download test dataset
    gdown.download(
        Config.PRIVATE_TEST_DATASET_FILE_PATH,
        str(Config.DATASET_PATH / "aicv115m_private_test.zip"),
    )

    with zipfile.ZipFile(str(Config.DATASET_PATH / "aicv115m_public_train.zip"), 'r') as z:
        z.extractall()
    shutil.move('aicv115m_public_train', str(Config.DATASET_PATH))

    with zipfile.ZipFile(str(Config.DATASET_PATH / "aicv115m_private_test.zip"), 'r') as z:
        z.extractall()
    shutil.move('aicv115m_private_test', str(Config.DATASET_PATH))

    with zipfile.ZipFile(str(Config.DATASET_PATH / "aicv115m_public_train/train_audio_files_8k.zip"), 'r') as z:
        z.extractall()
    shutil.move('train_audio_files_8k', str(Config.DATASET_PATH / "aicv115m_public_train/"))