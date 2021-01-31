import numpy as np
from gensim.models import Word2Vec
from tensorflow.python.keras.utils.np_utils import to_categorical

from common.constants import WORD2VEC_MODEL


def generate_word2vec_model(reviews):
    documents = [doc.text for i, doc in enumerate(reviews)]
    word2vec_model = Word2Vec(documents, size=100, window=3, min_count=1)
    word2vec_model.save(WORD2VEC_MODEL)

    return word2vec_model


def prepare_dataset_word2vec(model, train_reviews, test_reviews):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for review in train_reviews:
        if review.stars <= 2:
            x_train.append(compute_embedding(model, review.text))
            y_train.append(0)
        elif review.stars >= 4:
            x_train.append(compute_embedding(model, review.text))
            y_train.append(1)

    for review in test_reviews:
        if review.stars <= 2:
            x_test.append(compute_embedding(model, review.text))
            y_test.append(0)
        elif review.stars >= 4:
            x_test.append(compute_embedding(model, review.text))
            y_test.append(1)

    return np.array(x_train), np.array(x_test), np.array(to_categorical(y_train)), np.array(to_categorical(y_test))


def compute_embedding(model, text):
    words = filter(lambda word: word in model.wv.vocab.keys(), text)
    vector = map(lambda word: model[word], words)
    vector = np.mean(list(vector), axis=0)
    return vector