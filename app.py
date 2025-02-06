from flask import Flask, request, render_template
from PIL import Image
from transformers import pipeline

predict_des = pipeline("image-classification", model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")

app = Flask(__name__)

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", error='No file provided')

    if file:
        
        image = Image.open(file)
        results = pipe(image)
        sentence = results[0]['generated_text']
        keywords = ["leaf", "leaves"]
        if any(keyword in sentence for keyword in keywords):
            result = predict_des(image)
            prediction = "This plant has " + result[0]['label'] 
        else: 
            prediction =  "The image you provided is not leaf"
        return render_template('index.html', prediction_text=  prediction)


if __name__ == '__main__':
    app.run(debug=True)
