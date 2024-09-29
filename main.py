import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
model = tf.keras.models.load_model('image_classifier.h5')
img = cv.imread('frog image.jpg')
img = cv.resize(img, (32, 32))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  
img = img / 255.0  
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
index = np.argmax(prediction)
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
print(f'Predicted class: {class_names[index]}')
plt.imshow(cv.cvtColor(cv.imread('frog image.jpg'), cv.COLOR_BGR2RGB))
plt.title(f'Prediction: {class_names[index]}')
plt.show()