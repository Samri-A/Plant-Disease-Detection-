from flask import Flask, request, render_template
import pickle
import numpy as np
import tensorflow.keras.models as mode
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image



app = Flask(__name__)


model = mode.load_model('cnn_model.h5')
label_binarizer = pickle.load(open("label_transform.pkl", "rb"))

default_image_size = tuple((256, 256))



@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


def convert_image_to_array(file_stream):
    image = Image.open(file_stream)
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(image, default_image_size)  # Resize the image to (256, 256)
    return img_to_array(image)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", error='No file provided')

    if file:
        image = convert_image_to_array(file.stream)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        disease_class = np.argmax(prediction[0])
        disease_label = label_binarizer.classes_[disease_class]
        return render_template('index.html', prediction_text=disease_label)


if __name__ == '__main__':
    app.run(debug=True)
