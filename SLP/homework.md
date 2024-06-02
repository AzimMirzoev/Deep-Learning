import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Step 1: Data Preparation

n_samples = 400
noise = 0.02
factor = 0.5
    #### use x_train (Feature vectors), y_train (Class ground truths) as training set
x_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    #### use x_test (Feature vectors) as test set
    #### you do not use y_test for this assignment.
x_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=factor)

    
# Assume X_train, y_train, X_test, y_test are your training and testing datasets

# Step 2: Model Definition
slp_classifier = Sequential()
slp_classifier.add(Dense(units=1, activation='sigmoid', input_dim=x_train.shape[1]))

mlp_classifier = Sequential()
mlp_classifier.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
mlp_classifier.add(Dense(units=64, activation='relu'))
mlp_classifier.add(Dense(units=1, activation='sigmoid'))

# Step 3: Model Compilation
slp_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

mlp_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Model Training
slp_classifier.fit(x_train, y_train, epochs=10, batch_size=32)

mlp_classifier.fit(x_train, y_train, epochs=10, batch_size=32)

# Step 5: Model Evaluation
slp_loss, slp_accuracy = slp_classifier.evaluate(x_test, y_test)

mlp_loss, mlp_accuracy = mlp_classifier.evaluate(x_test, y_test)

print("SLP Classifier - Loss: {}, Accuracy: {}".format(slp_loss, slp_accuracy))
print("MLP Classifier - Loss: {}, Accuracy: {}".format(mlp_loss, mlp_accuracy))

# Step 6: Result Visualization
# You can visualize the results as per your requirements

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='.')
plt.title("Train data distribution")
plt.show()


# Training data
```
Training data, also known as training dataset, refers to a set of examples used ot train a machine,
learning model. It oncsists of input data points along with their corresponding output labels or target
values. The machine learning modle learns pattersn and relationships from the training data model
 to learn and improve its performance. It should be representative of the real-world scenarios that the model will
encoutner during inference. 

```


# Keras and tf.keras

```
Keras is an open source deep learning library wirtten in Python.
Keras is popular because it made easier to use all the mathematical formuals of
tensorflow. Before, we must had used the Thean and PyTorch to make simple MLP.

```

```
import tensorflow as tf
model = tf.keras.Sequantial()
```

# Deep learning Model life-cycle

1. Define the model
2. Compiler the model
3. Fit the model
4. Evaluate the model
5. Make predictions


### 1) Defining the model

```
    Defining the model requires first selecting the type of model you need and then chosing the architecture or
network topology.


From an API perspective, this involes defining the layers of the model, configuring each layer with a numnber
of nodes and activation function, and connneting the layers together into a cohesive model.

!!! MODEL CAN BE DEFINED EAITHER WITH THE SEQUENTIAL API OR THE FUNCTIONAL API !!!

```

### 2) Compile the model

```
    Compiling the modle requires first selecting a loos function that you want to optimize such as mean squared
error or cross-entropy.

It also requires that you select and alogrithm to perform the optimization preocedure, typically SGD or ADAM.


From API persepctive, this involves calling a function to compile the model with chosen configuration which will
prepare approriate data structure required for the efficient use of the model you defined.


```

### 3) Fit the model

```
    Fitting the model requires that you first select the training configuration, such as the number of epochs and
the bach_size.

From API perspective, this involes calling a function to perform the training process.

```

### 4) Make a prediction

```
    Making a prediction is the final step in the life cydle. It is why we wanted the model in the first place. 

```



# Sequantial Mdel API (SIMPLE)

```
    It is refered as sequantial because it is the most easiest. You define your model one by one.

```

EX:

```
from tensorflow.keras import Sequantial
from tensorflow.keras.layers import Dense

#define the model
model = Sequantial()
model.add(Dense(10, input_shape=(8,))) # accepts 8 inputs
model.add(Dense(1))


```


# Functional model API

Complex but more flexible. In involves explicitly connecting the output of one layer to the input of another layer. Each connection is specified. 

Firs, an input layer muse be defined via the Input class, and the shape of an input sample is specified. We mst 
retain a reference to the input layer when defining the model. 

```
x_in = Input(shape=(8,))
```
Next a fully connectd layer can be conneced to the input by calling the layer and passing the input layer.

```
x = Dense(10)(x_in)
```

We can connect to the output layer with the same manner. 

```
x_out = Dense(1)(x)
```

Full example

```
fron tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers improt Dense

x_in = Input(shape=(8,))
x = Dense(10)(x_in)
x_out = Dense(1)(x)

model = Model(inputs = x_in, outputs=x_out)
```

