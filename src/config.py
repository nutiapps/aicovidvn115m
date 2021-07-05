from pathlib import Path
class Config:
    ASSETS_PATH = Path("./data")
    TRAIN_DATASET_FILE_PATH = "https://drive.google.com/uc?id=1MPhz3zYl2yefCq-J5XySbFJt99BfKIZD"
    PRIVATE_TEST_DATASET_FILE_PATH = "https://drive.google.com/uc?id=1hP8rHwJ_bz3J1T4MtEEp53ZBe9fdFKrW"
    ROOT_TRAIN_DIR = ASSETS_PATH / "raw/aicv115m_public_train"
    ROOT_TEST_DIR = ASSETS_PATH / "raw/aicv115m_private_test"
    DATASET_PATH = ASSETS_PATH / "raw"
    FEATURES_PATH = ASSETS_PATH / "features"
    MODELS_PATH = ASSETS_PATH / "models"
    METRICS_FILE_PATH = ASSETS_PATH / "scores.json"
    FI_FILE_PATH = ASSETS_PATH / "fi.json"
    PLOT_FILE_PATH = ASSETS_PATH / "plots.json"
    SUBMISSION_PATH = ASSETS_PATH / "subs"