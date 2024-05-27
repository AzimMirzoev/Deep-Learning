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
