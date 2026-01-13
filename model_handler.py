from transformers import MT5ForConditionalGeneration, AutoTokenizer

class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """Load model and tokenizer."""
        print("\n=== Loading Model ===")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.model = MT5ForConditionalGeneration.from_pretrained(
            self.config.MODEL_NAME
        ).to(self.config.DEVICE)
        
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        
        print(f"Model loaded on {self.config.DEVICE}")
    
    def preprocess(self, batch):
        """Tokenize input-target pairs."""
        inputs = ["next: " + t for t in batch["input_text"]]
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length"
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                batch["target_text"],
                max_length=self.config.MAX_TARGET_LENGTH,
                truncation=True,
                padding="max_length"
            )
        
        model_inputs["labels"] = [
            [t if t != self.tokenizer.pad_token_id else -100 for t in label]
            for label in labels["input_ids"]
        ]
        
        return model_inputs
    
    def tokenize_datasets(self, train_ds, val_ds):
        """Apply tokenization to datasets."""
        train_tok = train_ds.map(
            self.preprocess, 
            batched=True, 
            remove_columns=train_ds.column_names
        )
        val_tok = val_ds.map(
            self.preprocess, 
            batched=True, 
            remove_columns=val_ds.column_names
        )
        return train_tok, val_tok
