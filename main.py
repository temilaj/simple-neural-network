import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

# we transpose it from a row to a column 4 X 1 matrix.
training_outputs = np.array([[0, 1, 1, 0]]).T

# seed the random values
np.random.seed(1)
# initialize weight by adding random values to the weight.
# create a 3 X 1 matix since we have three inputs and one output
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(1):

    input_layer = training_inputs
    output = sigmoid(np.dot(input_layer, synaptic_weights))

print('outputs after training: ')
print(output)
