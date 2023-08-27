from flask import Flask, request, jsonify
from keras.models import model_from_json
from pathlib import Path
import numpy as np
import tensorflow as tf
import io
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
CORS(app)

class_names = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

file = Path("./grape-leaf-disease-detection/cnn/model_structure.json")
cnn_model_structure = file.read_text()
cnn_model = model_from_json(cnn_model_structure)
cnn_model.load_weights("./grape-leaf-disease-detection/cnn/model_weights.h5")

svm_model = joblib.load("./grape-leaf-disease-detection/svm/svm_model.joblib")

@app.route('/predict_cnn', methods=['POST'])
def predict_cnn():
    try:
        file = request.files['image']
        img_bytes = file.read()
        img = tf.keras.utils.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        image_to_test = tf.keras.utils.img_to_array(img)
        list_of_images = np.expand_dims(image_to_test, axis=0)
        results = cnn_model.predict(list_of_images)
        single_result = results[0]
        most_likely_class_index = int(np.argmax(single_result))
        class_label = class_names[most_likely_class_index]
        response = {
            "predicted_class": class_label,
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    try:
        file = request.files['image']
        img_bytes = file.read()
        img = tf.keras.utils.load_img(io.BytesIO(img_bytes), target_size=(32, 32))  # Adjust target size
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        img_flatten = img_array.reshape(1, -1)
        svm_prediction = svm_model.predict(img_flatten)
        response = {
            "predicted_class": svm_prediction[0],
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
