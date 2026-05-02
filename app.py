from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("model.h5")

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        img_path=None,
        explanation=None
    )

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return render_template(
            "index.html",
            prediction=None,
            img_path=None,
            explanation=None
        )

    file = request.files["file"]

    if file.filename == "":
        return render_template(
            "index.html",
            prediction=None,
            img_path=None,
            explanation=None
        )

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # model prediction
    result = model.predict(img)
    score = float(result[0][0])

    if score > 0.5:
        prediction = "No Lung Cancer Detected"
        explanation = "The uploaded CT scan image appears normal. No suspicious abnormalities were identified."
    else:
        prediction = "Lung Cancer Detected"
        explanation = "The AI model found suspicious abnormal patterns in the uploaded CT scan image. Please consult a doctor."

    return render_template(
        "index.html",
        prediction=prediction,
        img_path=filepath,
        explanation=explanation
    )

if __name__ == "__main__":
    app.run(debug=True)