from config import Config
from data_loader import DataLoader
from data_processor import DataProcessor
from model_handler import ModelHandler
from trainer import KhmerTrainer
from predictor import Predictor

def main():
    # Initialize configuration
    config = Config()
    
    # Load and clean data
    data_loader = DataLoader(config.DATASET_PATH, config.FILES)
    cleaned_lines = data_loader.load_and_clean(min_length=config.MIN_LINE_LENGTH)
    
    # Create datasets
    data_processor = DataProcessor(config)
    train_ds, val_ds = data_processor.create_dataset(cleaned_lines)
    
    # Load model and tokenizer
    model_handler = ModelHandler(config)
    model_handler.load_model()
    
    # Tokenize datasets
    train_tok, val_tok = model_handler.tokenize_datasets(train_ds, val_ds)
    
    # Setup and run training
    trainer = KhmerTrainer(config, model_handler)
    trainer.setup_trainer(train_tok, val_tok)
    trainer.train()
    trainer.save()
    
    # Test predictions
    predictor = Predictor(model_handler)
    test_inputs = [
        "សម្តេចពុក",
        "ការអប់រំ",
        "ប្រទេសកម្ពុជា"
    ]
    predictor.test_predictions(test_inputs)
    
    print("\n Khmer Next Word Prediction Ready")

if __name__ == "__main__":
    main()