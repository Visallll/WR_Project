import time
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from .schemas import SuggestRequest, SuggestResponse
from .predictor import predict_top_k
from .model_handler import logger

app = FastAPI(
    title="Khmer GPT Text Prediction API",
    version="1.0"
)

templates = Jinja2Templates(directory="templates")

@app.get("/")
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    start = time.time()

    suggestions = predict_top_k(req.text, req.top_k)

    latency = round(time.time() - start, 4)
    logger.info(f"text='{req.text}' | latency={latency}s")

    return {
        "input_text": req.text,
        "suggestions": suggestions
    }
