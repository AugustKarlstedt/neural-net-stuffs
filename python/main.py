import numpy as np

XOR_INPUTS = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

XOR_TARGETS = np.array([
    [0],
    [1],
    [1],
    [0],
])

class neural_network:

  def __init__(self):
    self.hidden_layer_nodes = [3]
    self.epochs = 10000
    self.mini_batch_size = 2
    self.eta = 0.1
    self.weights = None
    self.biases = None
    self.layer_node_counts = None
    self.layer_count = None

  @staticmethod
  def sigmoid(z: np.array):
    return 1.0 / (1.0 + np.exp(-z))

  @staticmethod  
  def sigmoid_prime(z: np.array):
    s = neural_network.sigmoid(z)
    return s * (1 - s)

  @staticmethod
  def relu(z: np.array):
    return np.maximum(0, z)

  @staticmethod
  def relu_prime(z: np.array):
    return 0 if z <= 0 else 1

  def forward(self, x: np.array):
    activations = [x]
    zs = []

    for l in range(self.layer_count-1):
      z = np.dot(self.weights[l], activations[l]) + self.biases[l]
      zs.append(z)
      a = neural_network.sigmoid(z)
      activations.append(a)

    return activations, zs

  def fit(self, x: list, y: list):
    self.layer_node_counts = [len(x[0])] + self.hidden_layer_nodes + [len(y[0])]
    self.layer_count = len(self.layer_node_counts)

    self.weights = [np.random.normal(0.0, 1.0, (self.layer_node_counts[i], self.layer_node_counts[i-1])) for i in range(1, self.layer_count)]
    self.biases = [np.random.normal(0.0, 1.0, (self.layer_node_counts[i], 1)) for i in range(1, self.layer_count)]

    for epoch in range(self.epochs):
      mini_batch_indices = np.random.randint(0, len(x), self.mini_batch_size)
      mini_batch_x = x[mini_batch_indices]
      mini_batch_y = y[mini_batch_indices]

      weight_deltas = [np.zeros((self.layer_node_counts[i], self.layer_node_counts[i-1])) for i in range(1, self.layer_count)]
      bias_deltas = [np.zeros((self.layer_node_counts[i], 1)) for i in range(1, self.layer_count)]

      for example_index in range(self.mini_batch_size):
        activations, zs = self.forward(mini_batch_x[example_index])

        cost = activations[-1] - mini_batch_y[example_index]
        delta = cost * neural_network.sigmoid_prime(zs[-1])

        weight_deltas[-1] = np.dot(delta, np.transpose(activations[-2]))
        bias_deltas[-1] = delta

        for l in range(2, self.layer_count):
          delta = np.dot(np.transpose(self.weights[-l+1]), delta) * neural_network.sigmoid_prime(zs[-l])
          weight_deltas[-l] = weight_deltas[-l] + np.dot(delta, np.transpose(activations[-l-1]))
          bias_deltas[-l] = bias_deltas[-l] + delta

      for l in range(self.layer_count-1):
        self.weights[l] = self.weights[l] - (self.eta / self.mini_batch_size) * weight_deltas[l]
        self.biases[l] = self.biases[l] - (self.eta / self.mini_batch_size) * bias_deltas[l]

      cost = 0.0
      n = len(x)
      for i in range(n):
        activations, zs = self.forward(x[i])
        cost += np.sum(np.nan_to_num(-y[i]*np.log(activations[-1])-(1-y[i])*np.log(1-activations[-1]))) / n

      print('Epoch: {} Cost: {}'.format(epoch, cost))

nn = neural_network()
nn.fit(XOR_INPUTS, XOR_TARGETS)
