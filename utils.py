import re
import torch
from PIL import Image

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    """
    Basic cleaning for Urdu/English text
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove links
    text = re.sub(r"@\w+", "", text)         # remove mentions
    text = re.sub(r"#", "", text)            # remove hashtags symbol
    text = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s]", "", text)  # keep Urdu + English
    text = text.strip()
    return text


# =========================
# LABEL MAPPING (BINARY)
# =========================
binary_labels = {
    0: "Safe",
    1: "Hate Speech"
}


# =========================
# LABEL MAPPING (MULTICLASS)
# =========================
multi_labels = {
    0: "Safe",
    1: "Offensive",
    2: "Hate Speech",
    3: "Abusive"
}


def decode_binary(pred):
    return binary_labels.get(int(pred), "Unknown")


def decode_multi(pred):
    return multi_labels.get(int(pred), "Unknown")


# =========================
# SOFTMAX HELPER (optional)
# =========================
def softmax(x):
    return torch.nn.functional.softmax(x, dim=-1)


# =========================
# IMAGE LOADER (optional helper)
# =========================
def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


# =========================
# DEBUG HELPER
# =========================
def print_model_output(name, output):
    print(f"{name} Output: {output}")