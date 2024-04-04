from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import io

app = Flask(__name__)
CORS(app)

model1 = load_model("model1.h5")
model2 = load_model("model2.h5")
model3 = load_model("model3.h5")
models = [model1, model2, model3]

class_names = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_bytes = io.BytesIO(file.read())

    img = image.load_img(file_bytes, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    def ensemble_predictions(models, testX):
        results = [model.predict(testX) for model in models]
        results = np.array(results)
        summed = np.sum(results, axis=0)
        return np.argmax(summed, axis=1)

    ensemble_pred = ensemble_predictions(models, img)

    predicted_class = class_names[ensemble_pred[0]]

    response = {
        'predicted_class': predicted_class
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
