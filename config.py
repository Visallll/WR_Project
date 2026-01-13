import torch

class Config:
    # Data paths
    DATASET_PATH = "/content/drive/MyDrive/dataset_for_spellcheck"
    FILES = ["kh_oscars_Dataset.txt", "​kh_CC100.txt"]
    
    # Model settings
    MODEL_NAME = "google/mt5-small"
    OUTPUT_DIR = "./khmer_nextword_mt5"
    
    # Training parameters
    TEST_LINES = 4000
    EPOCHS = 3
    BATCH_SIZE = 16
    LR = 2e-4
    
    # Data processing
    MIN_PREFIX = 8
    MAX_PREFIX = 48
    TARGET_CHARS = 6
    STRIDE = 2
    MIN_LINE_LENGTH = 12
    
    # Tokenization
    MAX_INPUT_LENGTH = 64
    MAX_TARGET_LENGTH = 8
    
    # Training
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    LOGGING_STEPS = 100
    SAVE_TOTAL_LIMIT = 2
    TEST_SPLIT = 0.1
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"