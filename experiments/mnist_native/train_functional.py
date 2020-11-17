"""
Functional API
https://www.tensorflow.org/guide/keras/functional
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs  = keras.Input(shape=(28, 28, 1))
x       = layers.Conv2D(32, 3, activation='relu')(inputs)
x       = layers.Flatten(input_shape=(28, 28))(x)
x       = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
y_train = y_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
y_test = y_test[..., tf.newaxis].astype("float32")

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)
model.save("my_functional_model")

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
