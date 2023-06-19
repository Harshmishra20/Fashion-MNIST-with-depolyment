# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:36:15 2023

@author: Dell
"""




import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Allow files with extension png, jpg, and jpeg
EXTENSIONS = ['png', 'jpg', 'jpeg']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONS


def init():
    global model
    model = load_model(r'D:\Data Science\Projects\Fashion Mnist Project\Clothing_apparel_prediction_web_app\clothing_classification_model.h5')


# Function to load and prepare the image in the right shape
def read_image(filename):
    # Load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # Convert the image to array
    img = img_to_array(img)
    # Reshape the image into a sample of 1 channel
    img = img.reshape(1, 28, 28, 1)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        try:
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('static', filename)
                file.save(file_path)
                img = read_image(file_path)
                # Predict the class of an image
                class_prediction = model.predict_classes(img)
                print(class_prediction)

                # Map apparel category with the numerical class
                apparel_categories = [
                    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
                ]
                product = apparel_categories[class_prediction[0]]

                return render_template('predict.html', product=product, user_image=file_path)
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."

    return render_template('predict.html')


if __name__ == "__main__":
    init()
    app.run()
