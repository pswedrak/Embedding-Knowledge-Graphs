from gensim.models import Doc2Vec, Word2Vec
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from common.constants import DOC2VEC_MODEL, REVIEW_TOKENS_PATH, REVIEW_TEST_TOKENS_PATH, SIMON_MODEL_TRAIN, \
    SIMON_MODEL_TEST, WORD2VEC_MODEL
from doc2vec.doc2vec import prepare_dataset_doc2vec
from sentiment_analysis.simon import prepare_dataset_simon
from text_processing.yelp_utils import read_reviews
from gensim_vectors.gensim_vectors import prepare_dataset
import numpy as np

from word2vec.word2vec import prepare_dataset_word2vec


def main():
    # evaluate_doc2vec()
    # evaluate_word2vec_pre_trained()
    # evaluate_wordvec()
    evaluate_wordvec_simon()
    # evaluate_simon()
    # evaluate_glove()
    # evaluate_doc2vec_simon()
    # evaluate_glove_simon()
    # evaluate_word2vec_simon()


def evaluate_simon():
    size = 100
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset_simon(train_reviews, test_reviews,
                                                             SIMON_MODEL_TRAIN, SIMON_MODEL_TEST)

    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_predicting_model(size)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_acc.append(scores[1])

    print('SIMON: Accuracy on the training data: {}% '.format(np.mean(training_acc)))
    print('SIMON: Accuracy on the test data: {}% '.format(np.mean(test_acc)))


def evaluate_word2vec_pre_trained():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset(train_reviews, test_reviews, "word2vec-google-news-300")
    dim = 300

    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_predicting_model(dim)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_acc.append(scores[1])

    print('WORD2VEC: Accuracy on the training data: {}% '.format(np.mean(training_acc)))
    print('WORD2VEC: Accuracy on the test data: {}% '.format(np.mean(test_acc)))


def evaluate_glove():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset(train_reviews, test_reviews, "glove-twitter-100")
    dim = 100

    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_predicting_model(dim)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_acc.append(scores[1])

    print('GloVe: Accuracy on the training data: {} '.format(np.mean(training_acc)))
    print('GloVe: Accuracy on the test data: {} '.format(np.mean(test_acc)))


def define_predicting_model(input_dim=100):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=input_dim))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    return model


def evaluate_doc2vec():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    doc2vec_model = Doc2Vec.load(DOC2VEC_MODEL)
    x_train, x_test, y_train, y_test = prepare_dataset_doc2vec(doc2vec_model, train_reviews, test_reviews)

    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_predicting_model()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_acc.append(scores[1])

    print('DOC2VEC: Accuracy on the training data: {}% '.format(np.mean(training_acc)))
    print('DOC2VEC: Accuracy on the test data: {}% '.format(np.mean(test_acc)))


def evaluate_doc2vec_simon():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    doc2vec_model = Doc2Vec.load(DOC2VEC_MODEL)
    x_train, x_test, y_train, y_test = prepare_dataset_doc2vec(doc2vec_model, train_reviews, test_reviews)
    x_train2, x_test2, y_train2, y_test2 = prepare_dataset_simon(train_reviews, test_reviews,
                                                             SIMON_MODEL_TRAIN, SIMON_MODEL_TEST)

    x_train_concat = np.zeros((len(x_train), x_train.shape[1]*2))
    x_test_concat = np.zeros((len(x_test), x_test.shape[1]*2))

    for i in range(len(x_train)):
        x_train_concat[i] = np.concatenate((x_train[i], x_train2[i]))
        assert y_train[i][0] == y_train2[i][0]

    for i in range(len(x_test)):
        x_test_concat[i] = np.concatenate((x_test[i], x_test2[i]))
        assert y_test[i][0] == y_test2[i][0]

    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_predicting_model(200)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train_concat, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test_concat, y_test, verbose=1)
        test_acc.append(scores[1])

    print('DOC2VEC & SIMON: Accuracy on the training data: {}% '.format(np.mean(training_acc)))
    print('DOC2VEC & SIMON: Accuracy on the test data: {}% '.format(np.mean(test_acc)))


def evaluate_glove_simon():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset(train_reviews, test_reviews, "glove-twitter-100")
    x_train2, x_test2, y_train2, y_test2 = prepare_dataset_simon(train_reviews, test_reviews,
                                                             SIMON_MODEL_TRAIN, SIMON_MODEL_TEST)

    x_train_concat = np.zeros((len(x_train), x_train.shape[1] + x_train2.shape[1]))
    x_test_concat = np.zeros((len(x_test), x_test.shape[1] + x_test2.shape[1]))

    for i in range(len(x_train)):
        x_train_concat[i] = np.concatenate((x_train[i], x_train2[i]))
        assert y_train[i][0] == y_train2[i][0]

    for i in range(len(x_test)):
        x_test_concat[i] = np.concatenate((x_test[i], x_test2[i]))
        assert y_test[i][0] == y_test2[i][0]

    dim = 200

    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_predicting_model(dim)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train_concat, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test_concat, y_test, verbose=1)
        test_acc.append(scores[1])

    print('GloVe & SIMON: Accuracy on the training data: {} '.format(np.mean(training_acc)))
    print('GloVe & SIMON: Accuracy on the test data: {} '.format(np.mean(test_acc)))


def evaluate_word2vec_simon():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset(train_reviews, test_reviews, "word2vec-google-news-300")
    x_train2, x_test2, y_train2, y_test2 = prepare_dataset_simon(train_reviews, test_reviews,
                                                             SIMON_MODEL_TRAIN, SIMON_MODEL_TEST)

    x_train_concat = np.zeros((len(x_train), x_train.shape[1] + x_train2.shape[1]))
    x_test_concat = np.zeros((len(x_test), x_test.shape[1] + x_test2.shape[1]))

    for i in range(len(x_train)):
        x_train_concat[i] = np.concatenate((x_train[i], x_train2[i]))
        assert y_train[i][0] == y_train2[i][0]

    for i in range(len(x_test)):
        x_test_concat[i] = np.concatenate((x_test[i], x_test2[i]))
        assert y_test[i][0] == y_test2[i][0]

    dim = 400
    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_predicting_model(dim)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train_concat, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test_concat, y_test, verbose=1)
        test_acc.append(scores[1])

    print('WORD2VEC & SIMON: Accuracy on the training data: {}% '.format(np.mean(training_acc)))
    print('WORD2VEC & SIMON: Accuracy on the test data: {}% '.format(np.mean(test_acc)))


def evaluate_wordvec():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    word2vec_model = Word2Vec.load(WORD2VEC_MODEL)
    x_train, x_test, y_train, y_test = prepare_dataset_word2vec(word2vec_model, train_reviews, test_reviews)

    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_predicting_model()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_acc.append(scores[1])

    print('WORD2VEC: Accuracy on the training data: {}% '.format(np.mean(training_acc)))
    print('WORD2VEC: Accuracy on the test data: {}% '.format(np.mean(test_acc)))


def evaluate_wordvec_simon():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    word2vec_model = Word2Vec.load(WORD2VEC_MODEL)
    x_train, x_test, y_train, y_test = prepare_dataset_word2vec(word2vec_model, train_reviews, test_reviews)
    x_train2, x_test2, y_train2, y_test2 = prepare_dataset_simon(train_reviews, test_reviews,
                                                             SIMON_MODEL_TRAIN, SIMON_MODEL_TEST)

    x_train_concat = np.zeros((len(x_train), x_train.shape[1] + x_train2.shape[1]))
    x_test_concat = np.zeros((len(x_test), x_test.shape[1] + x_test2.shape[1]))

    for i in range(len(x_train)):
        x_train_concat[i] = np.concatenate((x_train[i], x_train2[i]))
        assert y_train[i][0] == y_train2[i][0]

    for i in range(len(x_test)):
        x_test_concat[i] = np.concatenate((x_test[i], x_test2[i]))
        assert y_test[i][0] == y_test2[i][0]

    dim = 200
    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_predicting_model(dim)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train_concat, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test_concat, y_test, verbose=1)
        test_acc.append(scores[1])

    print('WORD2VEC & SIMON: Accuracy on the training data: {}% '.format(np.mean(training_acc)))
    print('WORD2VEC & SIMON: Accuracy on the test data: {}% '.format(np.mean(test_acc)))


if __name__ == "__main__":
    main()
