import pandas as pd
import tensorflow as tf
import numpy as np
import cv2 as cv  
import matplotlib.pyplot as plt

# Load MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=3)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)
print("Test Loss:", loss)

# Prediction on custom images
for x in range(1, 10):
    try:
        # Read and process the image
        img = cv.imread(f'{x}.png', cv.IMREAD_GRAYSCALE)  # Read as grayscale
        img = cv.resize(img, (28, 28))  # Resize to 28x28
        img = np.invert(img)  # Invert the image
        img = np.array([img])  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img)
        print("----------------")
        print("The predicted value is:", np.argmax(prediction))
        print("----------------")
        
        # Display the image
        plt.imshow(img[0], cmap='gray')
        plt.show()
    except Exception as e:
        print(f"Error processing image {x}: {e}")
