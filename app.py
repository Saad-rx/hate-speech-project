from flask import Flask, request, jsonify

from model import (
    predict_bert_binary,
    predict_bert_multi,
    predict_xlm_binary,
    predict_xlm_multi,
    predict_image_resnet,
    predict_image_clip
)

app = Flask(__name__)


# =========================
# TEXT API
# =========================
@app.route("/predict_text", methods=["POST"])
def predict_text():
    text = request.json.get("text")

    return jsonify({
        "bert_binary": predict_bert_binary(text),
        "bert_multiclass": predict_bert_multi(text),
        "xlm_binary": predict_xlm_binary(text),
        "xlm_multiclass": predict_xlm_multi(text)
    })


# =========================
# IMAGE API (RESNET + CLIP)
# =========================
@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files["image"]
    path = "temp.jpg"
    file.save(path)

    return jsonify({
        "resnet": predict_image_resnet(path),
        "clip": predict_image_clip(path)
    })


if __name__ == "__main__":
    app.run(debug=True)