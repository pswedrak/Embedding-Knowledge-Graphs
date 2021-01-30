import numpy as np
import gensim.downloader as api
from tensorflow.python.keras.utils.np_utils import to_categorical


def prepare_dataset(train_reviews, test_reviews, vectors_name):
    wv_from_bin = api.load(vectors_name)
    vocab = list(wv_from_bin.vocab.keys())
    print("Loaded vocab size %i" % len(vocab))

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for review in train_reviews:
        if review.stars <= 2:
            x_train.append(compute_embedding(wv_from_bin, review.text))
            y_train.append(0)
        elif review.stars >= 4:
            x_train.append(compute_embedding(wv_from_bin, review.text))
            y_train.append(1)

    for review in test_reviews:
        if review.stars <= 2:
            x_test.append(compute_embedding(wv_from_bin, review.text))
            y_test.append(0)
        elif review.stars >= 4:
            x_test.append(compute_embedding(wv_from_bin, review.text))
            y_test.append(1)

    return np.array(x_train), np.array(x_test), np.array(to_categorical(y_train)), np.array(to_categorical(y_test))


def compute_embedding(wv_from_bin, text):
    words = filter(lambda word: word in wv_from_bin.vocab.keys(), text)
    vector = map(lambda word: wv_from_bin[word], words)
    vector = np.mean(list(vector), axis=0)
    return vector

