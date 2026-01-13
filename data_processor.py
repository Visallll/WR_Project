import numpy as np
from typing import List, Dict, Optional
from datasets import Dataset

class DataProcessor:
    def __init__(self, config):
        self.config = config
    
    def create_next_word_pairs(
        self,
        lines: List[str],
        max_lines: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Create input-target pairs for next word prediction."""
        pairs = []
        
        if max_lines:
            lines = lines[:max_lines]
        
        for text in lines:
            L = len(text)
            if L < self.config.MIN_PREFIX + self.config.TARGET_CHARS:
                continue
            
            for p in range(
                self.config.MIN_PREFIX,
                min(L - self.config.TARGET_CHARS, self.config.MAX_PREFIX),
                self.config.STRIDE
            ):
                input_text = text[:p]
                target_text = text[p:p + self.config.TARGET_CHARS]
                
                pairs.append({
                    "input_text": input_text,
                    "target_text": target_text
                })
        
        return pairs
    
    def create_dataset(self, lines: List[str]) -> tuple:
        """Create train and validation datasets."""
        print("\n=== Creating Next Word Pairs ===")
        pairs = self.create_next_word_pairs(lines, max_lines=self.config.TEST_LINES)
        np.random.shuffle(pairs)
        print(f"✓ Total next-word samples: {len(pairs)}")
        
        dataset = Dataset.from_list(pairs)
        dataset = dataset.train_test_split(test_size=self.config.TEST_SPLIT, seed=42)
        
        train_ds = dataset["train"]
        val_ds = dataset["test"]
        print(f"✓ Train: {len(train_ds)} | Val: {len(val_ds)}")
        
        return train_ds, val_ds