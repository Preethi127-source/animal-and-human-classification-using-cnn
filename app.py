import flask as f
import tensorflow as tf
import numpy as np
import cv2 as cv

app = f.Flask(__name__)

model = tf.keras.models.load_model('image_classifier.h5')
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

@app.route('/')
def index():
    return f.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = f.request.files['file']
    img = cv.imdecode(np.frombuffer(file.read(), np.uint8), cv.IMREAD_COLOR)
    img = cv.resize(img, (32, 32))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    index = np.argmax(prediction)
    predicted_class = class_names[index]
    
    return f.jsonify({'class': predicted_class})

if __name__== '_main_':
    app.run(debug=True)