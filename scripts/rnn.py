import numpy as np 
from .functions import softmax, tanh


class RNNNumpy:
    def __init__(self, hidden_size, vocab_size,  bptt_truncate=4):
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
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # TODO: I don't know the meaning of this yet but it will be clear soon
        self.bptt_truncate = bptt_truncate
    
    def forward(self, x):
        """
        X denoting one training sample or a sentence
        
        Args:
            x ([type]): [description]
        """
        T = len(x)
        s = np.zeros((T, self.hidden_size)) # we are saving all s in an numpy array
        s[-1] = np.zeros(self.hidden_size) # we initialize it with zeros
        o = np.zeros((T, self.word_dim))
        for t in np.arange(T):
            s[t] = tanh(np.dot(self.U, x[t]) + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return o, s

    def predict(self, x):
        o, s = self.forward(x)
        return np.argmax(o, axis=1)

    def calculate_loss(self, y_predicted, y):
        log_loss = y * np.log(y_predicted)
        cross_entropy = -np.mean(log_loss)
        return cross_entropy

    def back_propagation_trought_time(self, x, y):
        T = len(y)
        o, s = self.forward(x)
        dl_dU = np.zeros(self.U.shape)
        dl_dV = np.zeros(self.V.shape)
        dl_dW = np.zeros(self.W.shape)
        delta_o = o 
        
        # TODO: should implement o-Y
        # seems to understand this but , I can improve it and make it readble
        delta_o[np.arrange(len(y)), y] -= 1
        for t in np.arrange(T):
            dl_dV += np.outer(delta_o[t], s[t])
            delta_t = self.V.T.dot(delta_o[t]) * (1-(np.power(s[t], 2)))

            # TODO this part is not well understood, will improve it
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dl_dW += np.outer(delta_t, s[bptt_step-1])
                dl_dU += np.outer(delta_t, x[t])
                delta_t = self.W.T.dot(delta_t) * (1-(np.power(s[bptt_step-1], 2)))
        return dl_dV, dl_dU, dl_dW
