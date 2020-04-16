from typing import List, Dict, Tuple
import csv
import numpy as np
import itertools
import nltk


def read_sentences_data(path):
    """
    Read csv data and at the given path
    Args:
        path ([type]): [description]
    """
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, skipinitialspace=True)
        sentences = itertools.chain(
            *[nltk.sent_tokenize(x[0].lower()) for x in reader])
        sentences = [
            f'{sentence_start_token} {sentence} {sentence_end_token}' for sentence in sentences]
        return sentences


def tokenize_sentence(sentences: List[str]) -> List[str]:
    """
    Given a list of sentences tokenized sentences into words

    Args:
        sentences ([type]): [description]
    """
    tokenize_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tokenize_sentences = list(filter(lambda x: len(x) > 3, tokenize_sentences))
    return tokenize_sentences


def get_word_frequencies(tokenized_sentences: List[str]) -> Dict[str, float]:
    """
    given tokenize sentence return word_frequencies

    Args:
        tokenized_sentences (List[str]): given tokenize_sentences

    Returns:
        Dict[str, float]: [description]
    """
    word_frequencies = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    return word_frequencies


def build_word_indexes(word_frequencies: Dict[str,
                                              float],
                       vocabulary_size: int,
                       unknown_token="UNKNOWN_TOKEN") -> Tuple[List[str],
                                                               Dict[int,
                                                                    str],
                                                               Dict[str,
                                                                    int]]:
    """
    create word to index dictionary and index to word dictionary
    Args:
        word_frequencies (Dict): dictionary of word frequencies
        vocabulary_size : our vocabulary size

    Return:
        dict ([type]): word to index and index to word
    """
    vocab = word_frequencies.most_common(vocabulary_size - 1)
    index_to_word = {index: word[0] for index, word in enumerate(vocab)}
    index_to_word[len(vocab)] = unknown_token
    word_to_index = {word: index for index, word in index_to_word.items()}
    return vocab, index_to_word, word_to_index


def replace_unknown_words(tokenized_sentences: List[str],
                          word_to_index: Dict[str,
                                              int],
                          unknown_token="UNKNOWN_TOKEN") -> List[str]:
    """
    Replace all words not in our vocabulary with the unknown token

    Args:
        tokenized_sentences (List[str]): [description]
        word_to_index (dict[str, int]): [description]

    Returns:
        List[str]: [description]
    """
    for index, sentence in enumerate(tokenized_sentences):
        tokenized_sentences[index] = [
            word if word in word_to_index.keys() else unknown_token for word in sentence]
    return tokenized_sentences


def create_training_data(
        tokenized_sentences: List[str], word_to_index: Dict[str, int]) -> Tuple[np.array, np.array]:
    """
    Create the training data

    Args:
        tokenized_sentences (List[str]): [description]
        word_to_index (dict[str, int]): [description]

    Returns:
        np.array: [description]
    """
    # x is is the a sentence without the last word
    X_train = np.asarray([[word_to_index[word] for word in sentence[:-1]]
                          for sentence in tokenized_sentences])
    # the corespondind y is the next word
    Y_train = np.asarray([[word_to_index[word] for word in sentence[1:]]
                          for sentence in tokenized_sentences])
    return X_train, Y_train


def get_sentence_data(path, vocabulary_size=8000):
    """
    get the sentence form the data we will be using

    Args:
        path ([type]): [description]
        vocabulary_size (int, optional): [description]. Defaults to 8000.
    """

    sentences = read_sentences_data(path)
    print(f'we are dealing with {len(sentences)} sentences')
    tokenize_sentences = tokenize_sentence(sentences)

    word_freq = get_word_frequencies(tokenize_sentences)

    print(f"Found {len(word_freq.items())} unique words tokens.")
    vocab, index_to_word, word_to_index = build_word_indexes(word_freq, vocabulary_size)

    print("Using vocabulary size {}.".format(vocabulary_size))
    print("The least frequent word in our vocabulary is '{}' and appeared {} times.".format(vocab[-1][0], vocab[-1][1]))

    tokenize_sentences = replace_unknown_words(tokenize_sentences, word_to_index)

    print("\nExample sentence: '{}'".format(sentences[1]))
    print("\nExample sentence after Pre-processing: '{}'\n".format(tokenize_sentences[0]))

    X_train, Y_train = create_training_data(tokenize_sentences, word_to_index)

    print("X_train shape: {}".format(str(X_train.shape)))
    print("y_train shape: {}".format(str(Y_train.shape)))

    # Print an training data example
    x_example, y_example = X_train[17], Y_train[17]
    print('x: \n {} \n {}'.format(" ".join([index_to_word[x] for x in x_example]), x_example))
    print('y: \n {} \n {}'.format(" ".join([index_to_word[x] for x in y_example]), y_example))
    return X_train, Y_train


if __name__ == "__main__":
    X_train, Y_train = get_sentence_data('data/raw/reddit-comments-2015-08.csv')
