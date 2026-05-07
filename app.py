from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

CORS(app)

# =========================
# GLOBAL MODEL VARIABLE
# =========================

model = None

# =========================
# MULTICLASS LABELS
# =========================

classes = [
    "good",
    "hole",
    "objects",
    "oil spot",
    "thread error"
]

# =========================
# LOAD MODEL
# =========================

def load_my_model():

    global model

    if model is None:

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(128,128,3),
            include_top=False,
            weights=None
        )

        base_model.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D()(
            base_model.output
        )

        x = tf.keras.layers.Dense(
            128,
            activation='relu'
        )(x)

        x = tf.keras.layers.Dropout(0.4)(x)

        output = tf.keras.layers.Dense(
            5,
            activation='softmax'
        )(x)

        model = tf.keras.Model(
            inputs=base_model.input,
            outputs=output
        )

        model.load_weights(
            "fabric_weights.weights.h5"
        )

    return model

# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(image):

    image = image.resize((128,128))

    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():

    return render_template("index.html")

# =========================
# PREDICTION ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "file" not in request.files:

            return jsonify({
                "error": "No file uploaded"
            })

        file = request.files["file"]

        image = Image.open(
            io.BytesIO(file.read())
        ).convert("RGB")

        processed = preprocess_image(image)

        loaded_model = load_my_model()

        prediction = loaded_model.predict(processed)

        predicted_class = classes[
            np.argmax(prediction)
        ]

        confidence = float(
            np.max(prediction)
        )

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )