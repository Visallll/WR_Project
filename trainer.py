import math
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"])
            metrics["perplexity"] = ppl
            print(f"Perplexity: {ppl:.4f}")

class KhmerTrainer:
    def __init__(self, config, model_handler):
        self.config = config
        self.model_handler = model_handler
        self.trainer = None
    
    def setup_trainer(self, train_tok, val_tok):
        """Setup Seq2Seq trainer."""
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            overwrite_output_dir=True,
            num_train_epochs=self.config.EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=32,
            learning_rate=self.config.LR,
            eval_strategy="steps",
            eval_steps=self.config.EVAL_STEPS,
            save_steps=self.config.SAVE_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_total_limit=self.config.SAVE_TOTAL_LIMIT,
            fp16=False,
            report_to="none",
            predict_with_generate=False
        )
        
        self.trainer = Seq2SeqTrainer(
            model=self.model_handler.model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            tokenizer=self.model_handler.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                self.model_handler.tokenizer, 
                self.model_handler.model
            )
        )
        
        self.trainer.add_callback(PerplexityCallback())
    
    def train(self):
        """Execute training."""
        print("\n Training Khmer Next Word Model...")
        self.trainer.train()
    
    def save(self):
        """Save model and tokenizer."""
        self.trainer.save_model(self.config.OUTPUT_DIR)
        self.model_handler.tokenizer.save_pretrained(self.config.OUTPUT_DIR)
        print(f"Model saved to {self.config.OUTPUT_DIR}")
