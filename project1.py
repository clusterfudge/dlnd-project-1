__author__ = "Sean Fitzgerald <sean@fitzgeralds.me>"
import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid, grapped from previous lessons
    """
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))

        self.learning_rate = learning_rate

        #### Set this to your implemented sigmoid function ####
        # TODO: Activation function is the sigmoid function
        self.activation_function = sigmoid

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs # output activation layer == f(x) = x

        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error
        # derivative of x == 1, so error for gradient descent is just difference
        output_errors = targets - final_outputs
        # TODO: Backpropagated error
        # errors propagated to the hidden layer
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)
        # hidden layer gradients
        hidden_grad = hidden_errors * hidden_outputs * (1 - hidden_outputs) # d(sigmoid(x))/dx

        # TODO: Update the weights
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.learning_rate * output_errors * hidden_outputs.T
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.learning_rate * np.dot(hidden_grad, inputs.T)

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs

        return final_outputs
