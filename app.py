from flask import Flask, render_template, request,url_for
import tensorflow as tf
from PIL import Image
import base64
import os


# Create Flask app
app = Flask(__name__)
def index():
    return render_template('index.html')

# Load the saved model
model = tf.keras.models.load_model('C:/Users/Gaurav/Desktop/Research/model.h5')

# Set the image size for model input
img_height, img_width = 224, 224

# Define the labels
labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Background_without_leaves",
    "Blueberry___healthy",
    "Cherry___healthy",
    "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___healthy",
    "Corn___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((img_height, img_width))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the request
    image = request.files['image']
    img = Image.open(image)

    # Preprocess the image
    processed_image = preprocess_image(img)

    # Make the prediction
    predictions = model.predict(processed_image)[0]
    predicted_label_index = tf.argmax(predictions)
    predicted_label = labels[predicted_label_index]

    static_dir = os.path.join(app.root_path, 'static')
    os.makedirs(static_dir, exist_ok=True)

    temp_image_path = os.path.join(static_dir, 'temp_image.jpg')
    processed_image = tf.keras.preprocessing.image.array_to_img(processed_image[0])
    processed_image.save(temp_image_path)
    image_path = 'temp_image.jpg'  # We only need the filename here, not the full path

    # Return the predicted label and image path to the index.html template
    return render_template('index.html', prediction=predicted_label, image_path=image_path)

# Run the Flask app
if __name__ == '__main__':
    os.makedirs(os.path.join(app.instance_path, 'static'), exist_ok=True)
    app.run()
