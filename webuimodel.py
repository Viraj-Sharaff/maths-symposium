from flask import Flask, request, render_template, jsonify
import numpy as np
import face_recognition
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('age_recognition_model.h5')

# Define class labels
class_labels = ['MIDDLE', 'YOUNG', 'OLD']

# Folder paths for images
woman_folder = 'woman'
man_folder = 'man'

def get_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            images.append(os.path.join(folder, filename))
    return images

def predict_age(image_path):
    input_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(input_image)
    if len(face_locations) == 0:
        return None
    face_images = [input_image[top:bottom, left:right] for top, right, bottom, left in face_locations]
    predictions = []
    for face_image in face_images:
        img_array = face_image / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        predictions.append(predicted_class_label)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test_model', methods=['GET', 'POST'])
def test_model():
    if request.method == 'POST':
        image_file = request.files['image']
        predictions = predict_age(image_file)
        return render_template('test_model.html', predictions=predictions)
    return render_template('test_model.html')

@app.route('/bias')
def bias():
    woman_images = get_images(woman_folder)
    man_images = get_images(man_folder)
    
    woman_correct = 0
    man_correct = 0
    
    # Count correct predictions for women
    for image_path in woman_images:
        predictions = predict_age(image_path)
        if predictions and 'MIDDLE' in predictions:
            woman_correct += 1
    
    # Count correct predictions for men
    for image_path in man_images:
        predictions = predict_age(image_path)
        if predictions and 'MIDDLE' in predictions:
            man_correct += 1
    
    # Calculate total counts
    total_woman = len(woman_images)
    total_man = len(man_images)
    
    # Pass variables to the template
    return render_template('bias.html', woman_correct=woman_correct, total_woman=total_woman,
                           man_correct=man_correct, total_man=total_man)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    man_images = get_images(man_folder)
    woman_images = get_images(woman_folder)
    man_correct = 0
    woman_correct = 0
    
    for image_path in man_images:
        predictions = predict_age(image_path)
        if predictions and 'MIDDLE' in predictions:
            man_correct += 1
    
    for image_path in woman_images:
        predictions = predict_age(image_path)
        if predictions and 'MIDDLE' in predictions:
            woman_correct += 1
    
    return jsonify({'man': man_correct, 'woman': woman_correct})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
