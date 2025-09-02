import numpy as np

class Perceptron:
    def __init__(self, number_of_inputs, number_of_outputs, learning_rate=0.3):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.learning_rate = learning_rate # A small number that controls how drastically we change the weights. It's a tuning parameter to prevent the learning from being too chaotic.
        self.weights = np.random.rand(number_of_outputs, number_of_inputs + 1) - 0.5         # This extra +1 in weight is the bias term (b): y = mx + b
        # self.weights = np.random.rand(number_of_outputs, number_of_inputs) * 0.01
        # self.biases = np.random.rand(number_of_outputs) * 0.01

    def _sigmoid_activation(self, u):
        return 1 / (1 + np.exp(-u))
    
    def train(self, inputs, expected_outputs):
        x = np.insert(inputs, 0, 1) # Insert the bias (1) term to the inputs vector at index 0
        
        # Calculate the network's output
        u = np.dot(self.weights, x)
        outputs = self._sigmoid_activation(u)
        
        # ___ This is where the actual "learning" happens. ___
        
        # Update weights based on the error - We calculate the error, which is simply the difference between the correct answer (expected_output) and the neuron's prediction (outputs). If the neuron predicted 0.8 but the answer was 1, the error is 0.2.
        error = expected_outputs - outputs
        
        """
        np.outer executa o produto externo entre dois vetores, resultando em uma matriz
        exemplo: np.outer([a, b], [c, d]) -> [[ac, ad], [bc, bd]]
        """
        weight_adjustment = np.outer(error, x, out=None) 
        self.weights += self.learning_rate * weight_adjustment
        
        return outputs
    
    
    def execute(self, inputs):
        # Add the bias to the inputs
        x = np.insert(inputs, 0, 1)
        
        # Calculate the network's output (without training)
        u = np.dot(self.weights, x)
        outputs = self._sigmoid_activation(u)
        
        return outputs
        
        