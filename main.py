import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import tensorflow as tf
import tensorflowjs as tfjs


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

accuracy, loss = model.evaluate(x_test, y_test)

print(accuracy)
print(loss)

# model.save('digits.model')

tfjs.converters.save_keras_model(model, 'digits.jsmodel')


# window = Tk()
# window.title("Python GUI App")
# window.configure(width=500, height=300)
# window.configure(bg='lightgray')
# window.mainloop()