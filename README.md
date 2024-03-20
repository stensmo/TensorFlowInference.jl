# TensorFlowInference.jl

Easy access to ready made TensorFlow models in Julia. Currently the focus is on running inference, loading a trained model is very easy, and use the inference results from Julia. Supports the latest TensorFlow versions (2.15), and can be made to support older versions, since the Julia code relies on the C API. You can use all of the TensorFlow C API which is huge, but there are convenience methods to get you started. 

Examples:
```
using TensorFlowInference
using MLDatasets 


testset = MNIST(:test)

testsetMatrix = testset.features


graph = TF_NewGraph();
status = TF_NewStatus();
saved_model_dir = "mnist/";


session = simpleLoadSessionFromSavedModel(saved_model_dir, graph, status)


printOutput(graph)

inputValues = testsetMatrix
graphInputName = "serving_default_flatten_1_input"
graphOutputName = "StatefulPartitionedCall"

outputMatrix = simpleSessionRun(session, graph, status, inputValues, graphInputName, graphOutputName)


```

To save a model, use the following Python example Code: 
```
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)



tf.saved_model.save(model, "/mydirectory/mnist")



```