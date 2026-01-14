from pydantic import BaseModel
from typing import List

class SuggestRequest(BaseModel):
    text: str
    top_k: int = 5

class Suggestion(BaseModel):
    word: str
    confidence: float

class SuggestResponse(BaseModel):
    input_text: str
    suggestions: List[Suggestion]
