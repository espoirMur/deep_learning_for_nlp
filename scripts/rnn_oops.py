import numpy as np
from .functions import softmax, sigmoid, tanh, log_loss


class Layer:
    def forward(self, x, s_prev, U, V, W):
        """

        compute the forward pass for the layer
        Args:
            x ([type]): [description]
            s_prev ([type]): [description]
        """
        self.s_in = np.add(np.matmul(x, U), np.matmul(W, s_prev))
        self.s_out = sigmoid(self.s_in)
        self.o_in = np.matmul(V, self.s_out)
        self.o_out = softmax(self.o_in)

    def backward(self, x, prev_s, y, U, V, W):
        """
        compute the backward propagation for the layer
        """
        self.loss = log_loss(self.o_out, y)
        self.dl_dq = 1  # where q = V*s_t or should put ones
        self.dq_ds = V
        self.dq_dv = self.s_out
        self.dso_dsi = tanh(self.s_out, derivate=True)
        self.dsi_du = x
        self.dsi_dw = prev_s
        self.dsi_dprev_s = W


class RNN:
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
        self.layers = list()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(self, x):
        """
        X denoting one training sample or a sentence
        
        Args:
            x ([type]): [description]
        """
        T = len(x)
        self.layers = []
        prev_s = np.zeros(self.hidden_size)
        for t in range(T):
            layer = Layer()
            input = np.zeros(self.vocab_size)
            input[x[t]] = 1 #I am not getting this 
            layer.forward(input, prev_s, self.U, self.V, self.W)
            prev_s = layer.s_out
            self.layers.append(layer)

    def calculate_gradients(self, x, y):
        """Calculate the gradient
        Not uderstanding well why going forward
        
        Args:
            x ([type]): [description]
            y ([type]): [description]
        """
        for t, layer in enumerate(self.layers):
            input = np.zeros(self.vocab_size)
            input[x[t]] = 1  # I am not getting this
            prev_s = np.zeros(self.hidden_size)
            layer.backward(input, prev_s, y[t], self.U, self.V, self.W)
            prev_s = layer.s_out

    def backward_gradient(self, x, y):
        """
        Back propagate the gradient backwar
        
        Args:
            x ([type]): [description]
            y ([type]): [description]
        """
        self.forward(x)
        self.calculate_gradients(x, y)
        dl_du = np.zeros_like(self.U.shape)
        dl_dv = np.zeros_like(self.V.shape)
        dl_dw = np.zeros_like(self.W.shape)
        T = len(self.layers)

        for t in np.arrange(T)[::-1]:
            layer = self.layers[t]
            dl_dv += np.outer(layer.dl_dq, layer.dq_dv)  # correct
            delta_t = np.matmul(layer.dl_dq, layer.dq_ds) * layer.dso_dsi
            
            # TODO: should find why we are doing this
            for i in np.arrange(max(0, t-T), t+1)[::-1]:
                delta_t = np.matmul(delta_t, self.layers[i].dsi_dprev_s) * self.layers[i].dso_dsi
                dl_du += np.outer(self.layers[i].dsi_du, delta_t)
                dl_dw += np.outer(self.layers[i].dsi_dw, delta_t)
        return (dl_du, dl_dw, dl_dv)

    def sgd_step(self, x, y, learning_rate):
        dU, dW, dV = self.backward(x, y)
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW
