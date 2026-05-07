from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

CORS(app)

# Load trained model
model = tf.keras.models.load_model(
    "fabric_model.keras",
    compile=False
)

# Classes
classes = ["good", "defective"]

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        })

    file = request.files["file"]

    image = Image.open(
        io.BytesIO(file.read())
    ).convert("RGB")

    processed = preprocess_image(image)

    prediction = model.predict(processed)

    predicted_class = classes[np.argmax(prediction)]

    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)