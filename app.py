from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# 🔐 API KEY from Render Environment
API_KEY = os.environ.get("API_KEY")

# Load model once (global)
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("model.h5", compile=False)
    return model

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
    try:
        # 🔐 API KEY CHECK
        key = request.headers.get("x-api-key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        # Load model
        model = load_model()

        # Get image
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file).convert("RGB")

        # Preprocess
        img = preprocess(image)

        # Prediction
        pred = model.predict(img)

        print("RAW PRED:", pred)

        # Output
        result = "Defective" if pred[0][0] > 0.5 else "Good"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)