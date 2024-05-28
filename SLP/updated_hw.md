import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import tensorflow as tf


# Making a MLP class

class MLP:
  def __init__(self, hidden_layer_conf, num_output_nodes):
    self.hidden_layer_conf = hidden_layer_conf
    self.num_output_nodes = num_output_nodes
    self.logic_op_model = None

  def build_model(self):
    input_layer = tf.keras.Input(shape=[2,])
    hidden_layers=input_layer

    if self.hidden_layer_conf is not None:
      for num_hidden_nodes in self.hidden_layer_conf:
        hidden_layers = tf.keras.layers.Dense(units=num_hidden_nodes, activation=tf.keras.activations.sigmoid, use_bias=True)(hidden_layers)


    output = tf.keras.layers.Dense(units=self.num_output_nodes, activation=tf.keras.activations.sigmoid, use_bias=True)(hidden_layers)

    self.logic_op_model = tf.keras.Model(inputs=input_layer, outputs=output)

    sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
    self.logic_op_model.compile(optimizer=sgd, loss="mse")

  def fit(self, x, y, batch_size, epochs):
    self.logic_op_model.fit(x=x,y=y,batch_size=batch_size, epochs=epochs)

  def predict(self, x, batch_size):
    prediction = self.logic_op_model.predict(x=x, batch_size=batch_size)
    return prediction



### SLP ###

class SLP:
    def __init__(self, num_output_nodes):
        self.num_output_nodes = num_output_nodes
        self.logic_op_model = None

    def build_model(self):
        input_layer = tf.keras.Input(shape=[2,])
        output = tf.keras.layers.Dense(units=self.num_output_nodes, activation=tf.keras.activations.sigmoid, use_bias=True)(input_layer)
        self.logic_op_model = tf.keras.Model(inputs=input_layer, outputs=output)

        sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.logic_op_model.compile(optimizer=sgd, loss="mse")

    def fit(self, x, y, batch_size, epochs):
        self.logic_op_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def predict(self, x, batch_size):
        prediction = self.logic_op_model.predict(x=x, batch_size=batch_size)
        return prediction



# Step 1: Data Preparation
def CircleClassify():
  n_samples = 400
  noise = 0.02
  factor = 0.5

  x_train, y_train = make_circles(n_samples=n_samples, noise = noise, factor=factor)

  x_test, y_test = make_circles(n_samples = n_samples, noise=noise, factor=factor)

  

# Step 6: Result Visualization
# You can visualize the results as per your requirements

  plt.scatter(x_train[:, 0], x_train[:, 1], c =y_train, marker='.')
  plt.title("Train data distribution")
  plt.show()

  ########## MLP ###########
  hidden_layer_conf = [10,5]
  mlp_classifier  = MLP(hidden_layer_conf = hidden_layer_conf, num_output_nodes = 1)
  mlp_classifier.build_model()
  mlp_classifier.fit(x = x_train, y = y_train, batch_size=1, epochs=30)
  mlp_prediction = mlp_classifier.predict(x=x_train, batch_size = 1)
  
  ###########################

  
  ########## SLP ###########
  
  slp_classifier = SLP(num_output_nodes=1)
  slp_classifier.build_model()
  slp_classifier.fit(x=x_train, y = y_train, batch_size=1, epochs=30)
  slp_prediction = slp_classifier.predict(x=x_train, batch_size=1)

  ###########################



  def plot_decision_boundary(prediction, X, y):
      h = 0.01
      x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
      y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
      xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
      Z = prediction(np.c_[xx.ravel(), yy.ravel()])
      Z = Z.reshape(xx.shape)
      plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
      plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')

  # Plot decision boundaries for SLP Classifier
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plot_decision_boundary(slp_prediction, x_test, y_test)
  plt.title('SLP Classifier Decision Boundary')

  # Plot decision boundaries for MLP Classifier
  plt.subplot(1, 2, 2)
  plot_decision_boundary(mlp_prediction, x_test, y_test)
  plt.title('MLP Classifier Decision Boundary')

  plt.show()


if __name__ == "__main__":
  CircleClassify()
