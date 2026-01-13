import re
from typing import List

class DataLoader:
    def __init__(self, dataset_path: str, files: List[str]):
        self.dataset_path = dataset_path
        self.files = files
    
    def load_files(self) -> List[str]:
        """Load all text files and return list of lines."""
        lines = []
        for fname in self.files:
            file_path = f"{self.dataset_path}/{fname}"
            with open(file_path, "r", encoding="utf-8") as f:
                file_lines = f.readlines()
                lines.extend(file_lines)
            print(f"Loaded {len(file_lines)} lines from {fname}")
        return lines
    
    @staticmethod
    def clean_khmer(text: str) -> str:
        """Clean Khmer text by removing unwanted characters."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(
            r"[^\u1780-\u17FF\u19E0-\u19FF\s.,!?;:'\"()\-ៗ់៎៏័៍៑្]",
            "",
            text
        )
        return text.strip()
    
    def load_and_clean(self, min_length: int = 12) -> List[str]:
        """Load files and clean the text."""
        print("\n=== Loading and Cleaning Data ===")
        lines = self.load_files()
        cleaned = [
            self.clean_khmer(line) 
            for line in lines 
            if len(self.clean_khmer(line)) > min_length
        ]
        print(f"Cleaned lines kept: {len(cleaned)}")
        return cleaned