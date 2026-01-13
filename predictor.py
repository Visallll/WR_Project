import torch
from typing import List

class Predictor:
    def __init__(self, model_handler):
        self.model_handler = model_handler
    
    def predict(self, text: str, max_length: int = 16, num_beams: int = 1) -> str:
        """Predict next word for given input text."""
        self.model_handler.model.eval()
        
        inputs = self.model_handler.tokenizer(
            "next: " + text,
            return_tensors="pt"
        ).to(self.model_handler.config.DEVICE)
        
        with torch.no_grad():
            output = self.model_handler.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams
            )
        
        prediction = self.model_handler.tokenizer.decode(
            output[0], 
            skip_special_tokens=True
        )
        return prediction
    
    def test_predictions(self, test_inputs: List[str]):
        """Test predictions on multiple inputs."""
        print("\n=== NEXT WORD PREDICTION TEST ===")
        for text in test_inputs:
            prediction = self.predict(text)
            print(f"\nInput: {text}")
            print(f"Next word: {prediction}")

