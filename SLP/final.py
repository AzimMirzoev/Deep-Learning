import numpy as np
import tensorflow as tf
from keras import Model 
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras import Input
from keras.activations import tanh


def MLP(n_features, x_train, y_train, x_test, epochs=300, batch_size=1):
  model = Sequential()
  model.add(Input(shape=(n_features)))
  model.add(Dense(5, activation='relu'))
  model.add(Dense(5, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
  model.compile(optimizer=sgd, loss='mse')

  model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

  prediction = model.predict(x_test, batch_size = batch_size)

  return prediction



def SLP(n_features, x_train, y_train, x_test, epochs=300, batch_size=1):
  model = Sequential()
  model.add(Input(shape=(n_features)))
  model.add(Dense(1, activation='softmax'))

  sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
  model.compile(optimizer=sgd, loss='mse')

  model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

  prediction = model.predict(x_test, batch_size = batch_size)

  return prediction


def solve():
  n_samples = 400
  noise = 0.02
  factor = 0.5
  x_train, y_train = make_circles(n_samples=n_samples, noise = noise, factor=factor)

  x_test, y_test = make_circles(n_samples = n_samples, noise=noise, factor=factor)

  slp_prediction = SLP(2, x_train, y_train, x_test, 30, 1)
  mlp_prediction = MLP(2, x_train, y_train, x_test, 30, 1)


  plt.figure(figsize = (10,5))
  plt.subplot(1,2,1)
  plt.scatter(x_test[:,0], x_test[:,1], c=slp_prediction, marker=".")
  plt.title("SLP prediction")
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')


  plt.subplot(1,2,2)
  plt.scatter(x_test[:,0], x_test[:,1], c=mlp_prediction, marker=".")
  plt.title("MLP prediction")
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')

  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  solve()
