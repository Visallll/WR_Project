import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "models/xglm-khmer"
WEIGHTS_PATH = f"{MODEL_DIR}/xglm_khmer_word.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load tokenizer EXACTLY like training
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# 🔹 Load base XGLM architecture
model = AutoModelForCausalLM.from_pretrained("facebook/xglm-564M")

# 🔹 Pad token fix (same as training)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# 🔹 Load fine-tuned weights
state_dict = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()
