import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical


def prepare_corpus_for_glove(corpusfile, outputfile):
    output = open(outputfile, 'w')
    with open(corpusfile) as file:
        tokens = file.readline()
        while len(tokens) > 0:
            tokens = tokens[1:-2]
            tokens = tokens.split(", ")
            tokens = list(map(lambda x: x[1:-1], tokens))
            for token in tokens:
                output.write(token)
                output.write(' ')
            output.write('\n')
            file.readline()
            tokens = file.readline()
    output.close()


def prepare_dataset_glove(glove_vectors_file, train_reviews, test_reviews):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    model = create_map_word_to_vector(glove_vectors_file)

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
    words = filter(lambda word: word in model.keys(), text)
    vector = map(lambda word: model[word], words)
    vector = np.mean(list(vector), axis=0)
    return vector


def create_map_word_to_vector(glove_vectors_file):
    model = {}
    with open(glove_vectors_file) as vectors_file:
        line = vectors_file.readline()
        while len(line) > 0:
            tokens = line.split(" ")
            tokens[-1] = tokens[-1][:-1]
            key = tokens[0]
            embedding = []
            for token in tokens[1:]:
                embedding.append(token)
            model[key] = np.array(embedding, dtype=float)
            line = vectors_file.readline()

    return model

