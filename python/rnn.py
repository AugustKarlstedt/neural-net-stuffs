import math
import numpy as np

# z is the result from tanh 
def tanh_derivative(z):
    return 1 - np.square(z)

sequence_dimensions = 4
input_dimensions = 1
hidden_dimensions = 16
output_dimensions = 1

alpha = 0.01
iterations = 20000

input_to_hidden_weights = np.random.normal(0.0, 1.0, [input_dimensions, hidden_dimensions])
hidden_to_output_weights = np.random.normal(0.0, 1.0, [hidden_dimensions, output_dimensions])
hidden_to_hidden_weights = np.random.normal(0.0, 1.0, [hidden_dimensions, hidden_dimensions])

input_to_hidden_deltas = np.zeros_like(input_to_hidden_weights)
hidden_to_output_deltas = np.zeros_like(hidden_to_output_weights)
hidden_to_hidden_deltas = np.zeros_like(hidden_to_hidden_weights)

for iteration in range(iterations + 1):

  # try to predict the next value in sine wave
  start = np.random.randint(16*np.pi)
  x = [math.sin(i) for i in range(start, start+sequence_dimensions)]
  y = [math.sin(start+sequence_dimensions)]

  # predicted values
  prediction = np.zeros_like(y)
  
  error = 0.0
  
  output_deltas = []
  hidden_values = [np.zeros(hidden_dimensions)]
    
  # forward
  for i in range(sequence_dimensions):
    # 
    X = np.array(x[i])
    Y = np.array(y)
    
    hidden = np.tanh(np.dot(X, input_to_hidden_weights) + np.dot(hidden_values[-1], hidden_to_hidden_weights))
    output = np.tanh(np.dot(hidden, hidden_to_output_weights))
    
    # once this is done looping
    # we get the last output
    # which is all we care about
    prediction = output

    errors = Y - output
    output_deltas.append(errors * tanh_derivative(output))
    hidden_values.append(hidden)
    
    error += np.abs(errors[0])

  if iteration % 100 == 0:
    print('Iteration:', iteration, 'Total Error:', error, 'Prediction:', prediction, 'Actual:', y)

  delta = np.zeros(hidden_dimensions)

  # backward
  for i in range(sequence_dimensions):
    X = np.array([x[i]])
    
    hidden = hidden_values[-i-1]
    prev_hidden = hidden_values[-i-2]
    
    output_delta = output_deltas[-i-1]
    hidden_delta = (np.dot(delta, np.transpose(hidden_to_hidden_weights)) + np.dot(output_delta, np.transpose(hidden_to_output_weights))) * tanh_derivative(hidden)
    
    hidden_to_output_deltas += np.dot(np.transpose(np.atleast_2d(hidden)), output_delta)
    hidden_to_hidden_deltas += np.dot(np.transpose(np.atleast_2d(prev_hidden)), hidden_delta)
    input_to_hidden_deltas += np.dot(np.transpose(X), hidden_delta)

    delta = hidden_delta 
  
  hidden_to_output_weights += alpha * hidden_to_output_deltas
  input_to_hidden_weights += alpha * input_to_hidden_deltas
  hidden_to_hidden_weights += alpha * hidden_to_hidden_deltas

  input_to_hidden_deltas *= 0.0
  hidden_to_output_deltas *= 0.0
  hidden_to_hidden_deltas *= 0.0