import numpy as np
from .functions import softmax, sigmoid, tanh


class Layer:
    # TODO : This need to be done on the RNN level
    def __init__(self, hidden_size, vocab_size):
        """
        function help us to return  initialize the neural networks parameters 
        for text prediction the the number of input and 
        the number of output is equals to the vocabulary size, as well
        
        Args:
            hidden_size ([type]): [description]
            vocab_size ([type]): [description]
        """
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            """
            Generate normal distribution but a lot can be done here 
            
            Args:
                shape ([type]): [description]
            
            Returns:
                [type]: [description]
            """
            return np.random.normal(scale=0.01, size=shape)
        
        self.U = normal((hidden_size, num_inputs))
        self.W = normal((hidden_size, hidden_size))
        # Output layer parameters
        self.V = normal((num_outputs, hidden_size))
        self.b_hidden = np.zeros((hidden_size, 1))
        self.b_out = np.zeros((vocab_size, 1))

    def forward(self, x, s_prev):
        """

        compute the forward pass for the layer
        Args:
            x ([type]): [description]
            s_prev ([type]): [description]
        """
        self.s_in = np.add(np.matmul(x, self.U), np.matmul(self.W, s_prev))
        self.s_out = sigmoid(self.s_in)
        self.o_in = np.matmul(self.V, self.s_out)
        self.o_out = softmax(self.o_in)

    def backward(self):
        """
        """
