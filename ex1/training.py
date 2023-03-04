import numpy as np

#sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

#derivatives of sigmoid function
def sigmoid_derivatives(x):
	return x * (1 - x)

#input dataset
training_inputs = np.array([[0,0,1],
			[1,1,1],
			[1,0,1],
			[0,1,1]])

#output dataset			
training_outputs = np.array([[0,1,1,0]]).T

#generate random weights
np.random.seed(1)

#initialize weights randomly with mean 0 to create weight matrix, synaptic weights 
synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(10000):
	
	#define input layer
	input_layer = training_inputs
	#product of the input layer with synaptic weights
	outputs = sigmoid(np.dot(input_layer, synaptic_weights))
	
	#error calculation
	error = training_outputs - outputs
	adjustments = error * sigmoid_derivatives(outputs)
	
	#update weights
	synaptic_weights += np.dot(input_layer.T,adjustments)
	
print('Synaptic weights after training')
print(synaptic_weights)

print('Output after training')
print(outputs)

