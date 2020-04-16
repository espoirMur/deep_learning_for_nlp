import numpy as np
from models.rnn import RNNumpy, train_with_sgd
from data.data_preprocessing import get_sentence_data

np.random.seed(10)

X_train, Y_train = get_sentence_data('data/raw/reddit-comments-2015-08.csv')
vocab_size = 8000
model = RNNumpy(vocab_size)

if __name__ == "__main__":
    losses = train_with_sgd(model,
                            X_train[:400],
                            Y_train[:400],
                            nepoch=10,
                            evaluate_loss_after=1)
