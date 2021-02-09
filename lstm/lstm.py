import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow.python.keras.utils.np_utils import to_categorical


def prepare_dataset_lstm(train_reviews, test_reviews):
    tokenizer = Tokenizer()
    texts = list(map(lambda x: x.text, train_reviews + test_reviews))
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(sequences)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    train_size = len(train_reviews)
    i = 0

    for sequence in sequences:
        if i < train_size:
            stars = train_reviews[i].stars
            if stars <= 2:
                x_train.append(np.array(sequence))
                y_train.append(0)
            elif stars >= 4:
                x_train.append(np.array(sequence))
                y_train.append(1)
        else:
            stars = test_reviews[i - train_size].stars
            if stars <= 2:
                x_test.append(np.array(sequence))
                y_test.append(0)
            elif stars >= 4:
                x_test.append(np.array(sequence))
                y_test.append(1)
        i += 1

    return np.array(x_train), np.array(x_test), np.array(to_categorical(y_train)), np.array(to_categorical(y_test)), tokenizer
