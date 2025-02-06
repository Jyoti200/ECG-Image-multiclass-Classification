from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the .keras model
model = load_model('ECG_dataset.keras')  # Ensure the model file is in the same directory as this script

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to 128x128 pixels
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 128, 128, 3)
    return image

# Define the route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file)
            processed_img = preprocess_image(img)
            prediction = model.predict(processed_img)
            predicted_class = np.argmax(prediction[0])  # Get the predicted class index

            # Map the predicted_class to actual class names
            classes = ['Normal', 'Covid-19', 'MI', 'MI_History', 'Abnormal Heartbeat']
            result = classes[predicted_class]

            return render_template('result.html', prediction=result)
    return render_template('index.html')

# Define the route for the result page
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
