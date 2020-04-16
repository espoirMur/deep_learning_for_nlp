import sys
import numpy as np
from .functions import softmax, tanh
from datetime import datetime


class RNNumpy:
    def __init__(self, hidden_size, vocab_size, bptt_truncate=4):
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
        # we are saving all s in an numpy array
        s = np.zeros((T, self.hidden_size))
        s[-1] = np.zeros(self.hidden_size)  # we initialize it with zeros
        o = np.zeros((T, self.word_dim))
        for t in np.arange(T):
            s[t] = tanh(np.dot(self.U, x[t]) + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return o, s

    def predict(self, x):
        o, s = self.forward(x)
        return np.argmax(o, axis=1)

    def calculate_loss(self, y_predicted, y):
        """
        this will calculate the loss for one training example
        we calculate the loss of y_1, y2,y3     and  be find the mean of it
        TODO : is this the right approach?
        """
        log_loss = y * np.log(y_predicted)
        return -np.mean(np.sum(log_loss, axis=1))

    def back_propagation_trough_time(self, x, y):
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
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (np.power(s[t], 2)))

            # TODO this part is not well understood, will improve it
            for bptt_step in np.arange(
                    max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dl_dW += np.outer(delta_t, s[bptt_step - 1])
                dl_dU += np.outer(delta_t, x[t])
                delta_t = self.W.T.dot(delta_t) * \
                    (1 - (np.power(s[bptt_step - 1], 2)))
        return dl_dV, dl_dU, dl_dW

    def numpy_sgd_step(self, x, y, learning_rate):
        dL_dV, dL_dU, dL_dW = self.back_propagation_trough_time(x, y)
        self.U -= learning_rate * dL_dU
        self.V -= learning_rate * dL_dV
        self.W -= learning_rate * dL_dW


def train_with_sgd(
        model,
        x_train,
        y_train,
        learning_rate=0.005,
        nepoch=100,
        evaluate_loss_after=5):
    """
    Train with sgd

    Args:
        model ([type]): [description]
        x_train ([type]): [description]
        y_train ([type]): [description]
        learning_rate (float, optional): [description]. Defaults to 0.005.
        nepoch (int, optional): [description]. Defaults to 100.
        evaluate_loss_after (int, optional): [description]. Defaults to 5.
    """
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(x_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(
                '{} loss after number of example seen = {} epoch = {}: {}'.format(
                    time, num_examples_seen, epoch, loss))

            # setting the learning rate if it's increasing
            if(len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print(f"setting learning rate to {learning_rate}")
            sys.stdout.flush()
    for i in range(len(y_train)):
        model.sgd_step(x_train[i], y_train[i], learning_rate)
        num_examples_seen += 1
