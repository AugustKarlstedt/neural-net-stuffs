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

def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
  s = sigmoid(z)
  return s * (1 - s)

def nn(x, y):
  np.random.seed(0x02171994)

  layer_node_counts = [len(x[0]), 2, len(y[0])]
  layer_count = len(layer_node_counts)

  epochs = 1000
  mini_batch_size = 2
  eta = 0.1

  weights = [np.random.normal(0.0, 1.0, (layer_node_counts[i], layer_node_counts[i-1])) for i in range(1, layer_count)]
  biases = [np.random.normal(0.0, 1.0, (layer_node_counts[i], 1)) for i in range(1, layer_count)]

  for epoch in range(epochs):
    mini_batch_indices = np.random.randint(0, len(x), mini_batch_size)
    mini_batch_x = x[mini_batch_indices]
    mini_batch_y = y[mini_batch_indices]

    weight_deltas = []
    bias_deltas = []

    for example_index in range(mini_batch_size):
      activations = [mini_batch_x[example_index]]
      zs = []

      for l in range(layer_count-1):
        z = np.dot(weights[l], activations[l]) + biases[l]
        zs.append(z)
        a = sigmoid(z)
        activations.append(a)

      cost = activations[-1] - mini_batch_y[example_index]
      delta = cost * sigmoid_prime(zs[-1])

      mini_batch_weight_deltas = []
      mini_batch_bias_deltas = [] 

      for l in range(layer_count-1, 1, -1):
        delta = np.dot(np.transpose(weights[l-1]), delta) * sigmoid_prime(zs[l-1]) 
        mini_batch_weight_deltas.insert(0, np.transpose(delta * activations[l-1]))
        mini_batch_bias_deltas.insert(0, delta)

      weight_deltas += mini_batch_weight_deltas
      bias_deltas += mini_batch_bias_deltas

    for l in range(layer_count-1):
      weights[l] = weights[l] - (eta / mini_batch_size) * weight_deltas[l]
      biases[l] = biases[l] - (eta / mini_batch_size) * bias_deltas[l]

    print('Epoch: {}'.format(epoch))

nn(XOR_INPUTS, XOR_TARGETS)
