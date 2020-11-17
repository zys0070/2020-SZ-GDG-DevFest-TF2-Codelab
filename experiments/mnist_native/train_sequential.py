"""
Sequential model
https://www.tensorflow.org/tutorials/quickstart/beginner
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
y_train = y_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
y_test = y_test[..., tf.newaxis].astype("float32")

model = keras.models.Sequential([
  layers.Conv2D(32, 3),# activation='relu'),
  layers.BatchNormalization(), # just for demonstration
  layers.ReLU(),
  layers.Flatten(input_shape=(28, 28)),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(10)
])

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

model.save('my_sequential_model')


# Export frozen graph (No need for OV2021, it can load SavedModel directly.)
#from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
#loaded = keras.models.load_model('my_sequential_model')
#concrete_func = loaded.signatures['serving_default']
#frozen_func = convert_variables_to_constants_v2(concrete_func)
#graph_def = frozen_func.graph.as_graph_def()
#with tf.io.gfile.GFile('my_sequential_model_frozen_graph.pb', 'wb') as f:
#   f.write(graph_def.SerializeToString())
