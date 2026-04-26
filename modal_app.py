import modal
import os
import io

# =========================================================
# 1. SETUP MODAL ENVIRONMENT
# =========================================================
app = modal.App("hate-speech-detection-api")
volume = modal.Volume.from_name("hate-speech-models", create_if_missing=True)

# Define the environment with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "transformers", "torchvision", "pillow", 
        "gdown", "requests", "fastapi", "python-multipart"
    )
    .pip_install("git+https://github.com/openai/CLIP.git")
)

# Your Google Drive File IDs
GD_FILE_IDS = {
    # Real Image Models
    "best_resnet50_multiclass.pth": "1u4AIMQ0B0bvoTBi6uSU886KtJuLb4vOF",
    "best_resnet50_binary.pth": "1U8ZPc6MP2VlfJO_Fm_yfXpsqjNxx_GLs",
    "best_clip_binary.pth": "1DtZ_rGnk_qMugCIOBAiilPtc5oVXi4SY",
    "best_clip_multiclass.pth": "1CKn3zkbpWuffPEqQMa3PAa5MctWlE0j4",
    
    # Dummy Text Models (Replace these IDs later)
    "bert_binary.pth": "DUMMY_BERT_BIN_ID",
    "bert_multi.pth": "DUMMY_BERT_MUL_ID",
    "xlm_binary.pth": "DUMMY_XLM_BIN_ID",
    "xlm_multi.pth": "DUMMY_XLM_MUL_ID",
}

# =========================================================
# 2. MODEL INFERENCE CLASS
# =========================================================
@app.cls(
    image=image,
    volumes={"/models": volume},
    gpu="T4",
    container_idle_timeout=300,
)
class HateSpeechModel:
    @modal.enter()
    def load_models(self):
        import torch
        from transformers import AutoTokenizer
        import clip
        from torchvision import models
        import torch.nn as nn
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Download Logic ---
        import gdown
        for filename, file_id in GD_FILE_IDS.items():
            path = f"/models/{filename}"
            if not os.path.exists(path) and "DUMMY" not in file_id:
                print(f"Downloading {filename}...")
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, path, quiet=False)
        volume.commit()

        # --- Load Tokenizers ---
        self.bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

        # --- Load Text Models (Wrapped in try/except for dummy IDs) ---
        self.bert_bin = self._try_load_model("/models/bert_binary.pth")
        self.bert_mul = self._try_load_model("/models/bert_multi.pth")
        self.xlm_bin = self._try_load_model("/models/xlm_binary.pth")
        self.xlm_mul = self._try_load_model("/models/xlm_multi.pth")

        # --- Load Image Models ---
        # ResNet Binary
        self.resnet_bin = models.resnet50(pretrained=False)
        self.resnet_bin.fc = nn.Linear(self.resnet_bin.fc.in_features, 2)
        if os.path.exists("/models/best_resnet50_binary.pth"):
            self.resnet_bin.load_state_dict(torch.load("/models/best_resnet50_binary.pth", map_location=self.device))
        self.resnet_bin.to(self.device).eval()

        # ResNet Multiclass
        self.resnet_mul = models.resnet50(pretrained=False)
        self.resnet_mul.fc = nn.Linear(self.resnet_mul.fc.in_features, 4)
        if os.path.exists("/models/best_resnet50_multiclass.pth"):
            self.resnet_mul.load_state_dict(torch.load("/models/best_resnet50_multiclass.pth", map_location=self.device))
        self.resnet_mul.to(self.device).eval()
        
        # CLIP setup
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Note: If your fine-tuned CLIP models are state_dicts, you would load them here like:
        # if os.path.exists("/models/best_clip_binary.pth"):
        #     self.clip_model.load_state_dict(torch.load("/models/best_clip_binary.pth"))

    def _try_load_model(self, path):
        import torch
        import os
        if os.path.exists(path):
            return torch.load(path, map_location=self.device).eval()
        return None

    def _clean_text(self, text):
        import re
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s]", "", text)
        return text.strip()

    @modal.method()
    def predict_text(self, text):
        import torch
        text = self._clean_text(text)
        
        b_in = self.bert_tok(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        x_in = self.xlm_tok(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        results = {}
        with torch.no_grad():
            # BERT Binary
            if self.bert_bin:
                results["bert_binary"] = "Hate Speech" if torch.argmax(self.bert_bin(**b_in).logits, dim=1).item() == 1 else "Safe"
            else:
                results["bert_binary"] = "Pending Model Upload"

            # BERT Multiclass
            if self.bert_mul:
                results["bert_multiclass"] = ["Safe", "Offensive", "Hate Speech", "Abusive"][torch.argmax(self.bert_mul(**b_in).logits, dim=1).item()]
            else:
                results["bert_multiclass"] = "Pending Model Upload"

            # XLM Binary
            if self.xlm_bin:
                results["xlm_binary"] = "Hate Speech" if torch.argmax(self.xlm_bin(**x_in).logits, dim=1).item() == 1 else "Safe"
            else:
                results["xlm_binary"] = "Pending Model Upload"

            # XLM Multiclass
            if self.xlm_mul:
                results["xlm_multiclass"] = ["Safe", "Offensive", "Hate Speech", "Abusive"][torch.argmax(self.xlm_mul(**x_in).logits, dim=1).item()]
            else:
                results["xlm_multiclass"] = "Pending Model Upload"

        return results

    @modal.method()
    def predict_image(self, image_bytes):
        import torch
        from PIL import Image
        from torchvision import transforms
        import clip
        
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # ResNet processing
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        res_in = transform(img).unsqueeze(0).to(self.device)
        
        # CLIP processing
        clip_in = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        labels = ["safe", "hate speech", "abusive", "neutral"]
        tokens = clip.tokenize(labels).to(self.device)

        with torch.no_grad():
            res_bin_out = torch.max(self.resnet_bin(res_in), 1)[1].item()
            res_mul_out = torch.max(self.resnet_mul(res_in), 1)[1].item()
            
            # Using zero-shot CLIP as fallback/base if custom weights aren't applied differently
            logits, _ = self.clip_model(clip_in, tokens)
            clip_out = labels[logits.argmax().item()]

        return {
            "resnet_binary": "Hate Speech" if res_bin_out == 1 else "Safe",
            "resnet_multiclass": ["Safe", "Offensive", "Hate Speech", "Abusive"][res_mul_out],
            "clip_prediction": clip_out
        }

# =========================================================
# 3. WEB ENDPOINTS
# =========================================================
import fastapi

web_app = fastapi.FastAPI()

@app.function()
@modal.web_endpoint(method="POST")
def api_predict_text(item: dict):
    model = HateSpeechModel()
    return model.predict_text.remote(item["text"])

@app.function()
@modal.web_endpoint(method="POST")
async def api_predict_image(image: fastapi.UploadFile):
    model = HateSpeechModel()
    img_bytes = await image.read()
    return model.predict_image.remote(img_bytes)
