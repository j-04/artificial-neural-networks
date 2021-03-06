import tensorflow as tf
import numpy as np
from PIL import Image

mnist = tf.keras.datasets.mnist

(training_data, training_labels), (test_data, test_labels) = mnist.load_data()
training_data, test_data = training_data / 255, test_data / 255

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=5)

model.evaluate(test_data, test_labels)

predictions = model.predict(test_data)
np.set_printoptions(suppress=True)

for i in range(15):
    print("Тестовое значение: ", test_labels[i])
    print("Значения выходного слоя нейронной сети: ", predictions[i])