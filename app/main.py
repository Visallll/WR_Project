import time
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from .schemas import SuggestRequest, SuggestResponse
from .predictor import predict_top_k
from .model_handler import logger

app = FastAPI(
    title="Khmer XGLM Text Prediction API",
    version="1.0"
)

templates = Jinja2Templates(directory="templates")

# =======================
# UI
# =======================
@app.get("/")
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# =======================
# MODEL PREDICTION
# =======================
@app.post(
    "/v1/models/xglm-khmer/predict",
    response_model=SuggestResponse
)
def predict_xglm(req: SuggestRequest):
    start = time.time()

    suggestions = predict_top_k(req.text, req.top_k)

    latency = round(time.time() - start, 4)
    logger.info(
        f"model=xglm-khmer | text='{req.text}' | latency={latency}s"
    )

    return {
        "input_text": req.text,
        "model": "xglm-khmer",
        "latency": latency,
        "suggestions": suggestions
    }
