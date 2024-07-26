import streamlit as st
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from PIL import Image
import numpy as np
import os

# Get the directory of the current script
working_dir = os.path.dirname(os.path.abspath(__file__))

 # working_dir will be '/home/user/project' if the script is located at '/home/user/project/load_model.py'
 # Full file path: /home/user/project/load_model.py
 # File name: load_model.py
 # Directory path: /home/user/project


# Construct the full path to the pre-trained model
model_path = f"{working_dir}/trained_model/trained_fashion_mnist_model.h5"

# Custom deserialization function
def custom_loss_deserialization(config):
    if config['class_name'] == 'SparseCategoricalCrossentropy':
        config['config'].pop('fn', None)  # Remove the unexpected argument
        return SparseCategoricalCrossentropy.from_config(config['config'])
    else:
        return tf.keras.losses.deserialize(config)

# Load the model with the custom deserialization function
with tf.keras.utils.custom_object_scope({'custom_loss_deserialization': custom_loss_deserialization}):
    model = tf.keras.models.load_model(model_path, custom_objects={'SparseCategoricalCrossentropy': custom_loss_deserialization})

# The model is now loaded and can be used for predictions, further training, or evaluation

# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Function to preprocess the uploaded image

   # Open the image file, resize it to 28x28 pixels, convert it to grayscale,
   # convert the image to a NumPy array and normalize pixel values to the range [0, 1],
   # then reshape the array to (1, 28, 28, 1) to include batch size and channel dimensions

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28,28))
    img = img.convert('L')
    img_array = np.array(img)/255.0
    img_array = img_array.reshape((1,28,28,1))
    return img_array

# Streamlit App
st.title('Fashion Item Classifier')

# '''
#     create a file uploader widget in the Streamlit app
#     The uploader allows users to upload an image file with the specified types
# '''

uploaded_image = st.file_uploader("Upload an image ...", type=["jpg","jpeg", "png"])

# Check if an image is uploaded
if uploaded_image is not None:
    image = Image.open(uploaded_image) # Open the uploaded image

    # Create two columns in the Streamlit layout
    col1, col2 =  st.columns(2)

    with col1:
        resized_image = image.resize((100,100)) # Resize the image to 100x100 pixels
        st.image(resized_image) # Display the resized image

    with col2:
        if st.button('clasify'): # Button for classification
            img_array = preprocess_image(uploaded_image)  # Preprocess the image for the model
            result = model.predict(img_array) # Predict the class using the model
            predicted_class = np.argmax(result)  # Get the index of the highest probability
            prediction = class_names[predicted_class] # Map index to class name
            st.success(f'Prediction: {prediction}')  # Display the prediction
