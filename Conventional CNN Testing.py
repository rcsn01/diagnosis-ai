import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
loaded_model = load_model('best_model.h5')

# Function to preprocess an image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to make a prediction
def make_prediction(image_path):
    img_array = preprocess_image(image_path)
    prediction = loaded_model.predict(img_array)
    if prediction[0][0] > prediction[0][1]:
        return 'Benign'
    else:
        return 'Malignant'

# Specify the path to the image you want to predict
image_path = "C:/Users/Deklin/Documents/UTS SHIT/Professional A/Dataset\Validate/benign/lungn3751.jpeg"  # Change this to the path of your image

# Make a prediction
result = make_prediction(image_path)
print('Prediction:', result)
