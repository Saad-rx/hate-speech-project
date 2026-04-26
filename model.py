import torch
from transformers import AutoTokenizer
from utils import clean_text, decode_binary, decode_multi

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TOKENIZERS
# =========================
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
xlm_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# =========================
# LOAD TEXT MODELS
# =========================
bert_binary_model = torch.load("saved_models/bert_binary.pth", map_location=device)
bert_multi_model = torch.load("saved_models/bert_multi.pth", map_location=device)

xlm_binary_model = torch.load("saved_models/xlm_binary.pth", map_location=device)
xlm_multi_model = torch.load("saved_models/xlm_multi.pth", map_location=device)

bert_binary_model.eval()
bert_multi_model.eval()
xlm_binary_model.eval()
xlm_multi_model.eval()

# =========================
# BERT BINARY
# =========================
def predict_bert_binary(text):
    text = clean_text(text)
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = bert_binary_model(**inputs)
        pred = torch.argmax(output.logits, dim=1).item()

    return decode_binary(pred)


# =========================
# BERT MULTICLASS
# =========================
def predict_bert_multi(text):
    text = clean_text(text)
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = bert_multi_model(**inputs)
        pred = torch.argmax(output.logits, dim=1).item()

    return decode_multi(pred)


# =========================
# XLM-R BINARY
# =========================
def predict_xlm_binary(text):
    text = clean_text(text)
    inputs = xlm_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = xlm_binary_model(**inputs)
        pred = torch.argmax(output.logits, dim=1).item()

    return decode_binary(pred)


# =========================
# XLM-R MULTICLASS
# =========================
def predict_xlm_multi(text):
    text = clean_text(text)
    inputs = xlm_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = xlm_multi_model(**inputs)
        pred = torch.argmax(output.logits, dim=1).item()

    return decode_multi(pred)


# =========================================================
# 🖼️ IMAGE MODELS (CLIP + RESNET ADDED HERE)
# =========================================================

from resnet_model import predict_resnet
from clip_model import predict_clip


def predict_image_resnet(image_path):
    return predict_resnet(image_path)


def predict_image_clip(image_path):
    labels = ["safe", "hate speech", "abusive", "neutral"]
    return predict_clip(image_path, labels)