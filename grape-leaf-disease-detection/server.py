from flask import Flask, request, jsonify
from keras.models import model_from_json
from pathlib import Path
import numpy as np
import tensorflow as tf
import io
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

class_names = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

file = Path("model_structure.json")
model_structure = file.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img_bytes = file.read()  # Read the uploaded image bytes
        img = tf.keras.utils.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        image_to_test = tf.keras.utils.img_to_array(img)
        list_of_images = np.expand_dims(image_to_test, axis=0)
        results = model.predict(list_of_images)
        single_result = results[0]
        most_likely_class_index = int(np.argmax(single_result))
        class_likelihood = single_result[most_likely_class_index]
        class_label = class_names[most_likely_class_index]
        response = {
            "class_label": class_label,
            "class_likelihood": float(class_likelihood)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

