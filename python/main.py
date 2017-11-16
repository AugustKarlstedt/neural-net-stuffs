import math
import numpy as np

iris = np.genfromtxt('iris.csv', delimiter=',')
IRIS_INPUTS, IRIS_TARGETS = np.split(iris, [4], axis=1)

IDENTITY_INPUTS = np.identity(4)
IDENTITY_TARGETS = IDENTITY_INPUTS

XOR_INPUTS = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]) # shape = (4, 2) where each row is an example

XOR_TARGETS = np.array([
    [0],
    [1],
    [1],
    [0],
]) # shape = (4, 1) where each row is an example

SIN_INPUTS = np.array([[x / 256] for x in range(256)])
SIN_TARGETS = np.array([[math.sin(x / 256)] for x in range(256)])

class neural_network:

  def __init__(self):
    self.hidden_layer_nodes = [32]
    self.epochs = 99999
    self.mini_batch_size = 32
    self.eta = 0.1
    self.momentum = 0.99
    self.weights = None
    self.biases = None
    self.layer_node_counts = None
    self.layer_count = None
    self.softmax_output = False

  @staticmethod
  def sigmoid(z: np.array):
    return 1.0 / (1.0 + np.exp(-z))

  @staticmethod  
  def sigmoid_prime(z: np.array):
    s = neural_network.sigmoid(z)
    return s * (1 - s)

  @staticmethod
  def tanh_prime(z: np.array):
    return 1 - np.square(np.tanh(z))

  @staticmethod
  def relu(z: np.array):
    return np.maximum(0, z)

  @staticmethod
  def relu_prime(z: np.array):
    return np.zeros(z.shape) if np.all(z <= 0) else np.ones(z.shape)

  @staticmethod
  def leaky_relu(z: np.array, leakiness=0.1):
    return np.maximum(leakiness * z, z)

  @staticmethod
  def leaky_relu_prime(z: np.array, leakiness=0.1):
    return np.full(z.shape, leakiness) if np.all(z <= 0) else np.ones(z.shape)

  @staticmethod
  def softmax(z: np.array):
    e = np.exp(z)
    return e / (np.sum(e) + np.finfo(float).eps)

  def forward(self, x: np.array):
    activations = [np.zeros((self.layer_node_counts[i])) for i in range(self.layer_count)]
    zs = [np.zeros((self.layer_node_counts[i])) for i in range(self.layer_count)]
    
    activations[0] = x
    zs[0] = x

    for l in range(self.layer_count-1):
      z = np.dot(self.weights[l], activations[l]) + self.biases[l]
      zs[l+1] = z
      if l == self.layer_count-2 and self.softmax_output: # last layer?
        a = neural_network.softmax(z)
      else:
        a = neural_network.sigmoid(z)
      activations[l+1] = a

    return activations, zs

  def backward(self, activations, zs, target):
    cost = activations[-1] - target
    delta = cost

    weight_nablas = [np.zeros((self.layer_node_counts[i], self.layer_node_counts[i-1])) for i in range(1, self.layer_count)]
    bias_nablas = [np.zeros(self.layer_node_counts[i]) for i in range(1, self.layer_count)]

    weight_nablas[-1] = np.dot(delta, activations[-1])
    bias_nablas[-1] = delta

    for l in range(2, self.layer_count):
      delta = np.dot(np.transpose(self.weights[-l+1]), delta)
      weight_nablas[-l] = np.dot(delta, np.transpose(activations[-l]))
      bias_nablas[-l] = delta

    return weight_nablas, bias_nablas

  def fit(self, x: list, y: list):
    n = len(x)
    mini_batch_count = math.ceil(n / self.mini_batch_size)
    self.layer_node_counts = [len(x[0])] + self.hidden_layer_nodes + [len(y[0])]
    self.layer_count = len(self.layer_node_counts)

    self.weights = [np.random.normal(0.0, 1.0, (self.layer_node_counts[i], self.layer_node_counts[i-1])) / np.sqrt(self.layer_node_counts[i-1]) for i in range(1, self.layer_count)]
    self.biases = [np.random.normal(0.0, 1.0, self.layer_node_counts[i]) for i in range(1, self.layer_count)]

    for epoch in range(self.epochs):
      mini_batches = np.array_split(np.random.choice(n, n, replace=False), mini_batch_count)

      weight_velocities = [np.zeros((self.layer_node_counts[i], self.layer_node_counts[i-1])) for i in range(1, self.layer_count)]
      bias_velocities = [np.zeros(self.layer_node_counts[i]) for i in range(1, self.layer_count)]

      for mini_batch_index in range(len(mini_batches)):
        weight_deltas = [np.zeros((self.layer_node_counts[i], self.layer_node_counts[i-1])) for i in range(1, self.layer_count)]
        bias_deltas = [np.zeros(self.layer_node_counts[i]) for i in range(1, self.layer_count)]

        mini_batch_indices = mini_batches[mini_batch_index]
        mini_batch_size = len(mini_batch_indices)
        mini_batch_x = x[mini_batch_indices]
        mini_batch_y = y[mini_batch_indices]

        for example_index in range(len(mini_batch_x)):
          activations, zs = self.forward(mini_batch_x[example_index])
          weight_nablas, bias_nablas = self.backward(activations, zs, mini_batch_y[example_index])

          for l in range(self.layer_count-1):
            weight_deltas[l] = weight_deltas[l] + weight_nablas[l]
            bias_deltas[l] = bias_deltas[l] + bias_nablas[l]
      
        for l in range(self.layer_count-1):
          weight_velocities[l] = self.momentum * weight_velocities[l] - (self.eta / mini_batch_size) * weight_deltas[l]
          bias_velocities[l] = self.momentum * bias_velocities[l] - (self.eta / mini_batch_size) * bias_deltas[l]
          self.weights[l] = self.weights[l] + weight_velocities[l]
          self.biases[l] = self.biases[l] + bias_velocities[l]

      cost = 0.0
      correct = 0
      for i in range(n):
        activations, zs = self.forward(x[i])
        if self.softmax_output:
          correct += np.all(np.argmax(activations[-1]) == np.argmax(y[i]))
        else:
          correct += np.all(np.abs(np.round(activations[-1] - y[i], 2)) <= 0.01)
        cost += np.sum(np.nan_to_num(-y[i]*np.log(activations[-1])-(1-y[i])*np.log(1-activations[-1])))

      print('Target: {} Output: {}'.format(y[-1], activations[-1]))

      print('Epoch: {} Cost: {:0.4f} Correct: {}/{}={:0.2f}%'.format(epoch, cost / n, correct, n, correct/n*100.0))


inputs = SIN_INPUTS
targets = SIN_TARGETS

nn = neural_network()
nn.fit(inputs, targets)