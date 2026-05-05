from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# 🔐 Get API key from environment
API_KEY = os.environ.get("API_KEY")

model = None

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return "Fabric Defect Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    global model

    # 🔐 API KEY CHECK
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    if model is None:
        model = tf.keras.models.load_model("model.keras")

    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    img = preprocess(image)
    pred = model.predict(img)

    result = "Defective" if pred[0][0] > 0.5 else "Good"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)