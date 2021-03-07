from gensim.models import Doc2Vec, Word2Vec
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.backend import get_value
from tensorflow.python.keras.layers import Dense, Embedding, SpatialDropout1D, LSTM
from common.constants import DOC2VEC_MODEL, REVIEW_TOKENS_PATH, REVIEW_TEST_TOKENS_PATH, SIMON_MODEL_TRAIN, \
    SIMON_MODEL_TEST, WORD2VEC_MODEL, GLOVE_VECTORS, WORD2VEC_MODEL_THREE_CLASSES
from doc2vec.doc2vec import prepare_dataset_doc2vec
from glove.glove import prepare_dataset_glove
from lstm.lstm import prepare_dataset_lstm
from sentiment_analysis.simon import prepare_dataset_simon
from text_processing.yelp_utils import read_reviews
from gensim_vectors.gensim_vectors import prepare_dataset
import numpy as np

from word2vec.word2vec import prepare_dataset_word2vec


def main():
    # evaluate_doc2vec()
    # evaluate_word2vec_pre_trained()
    # evaluate_wordvec(True)
    # evaluate_wordvec_simon()
    evaluate_simon()
    # evaluate_glove_pretrained()
    # evaluate_doc2vec_simon()
    # evaluate_glove_pretrained_simon()
    # evaluate_word2vec_simon()
    # evaluate_glove()
    # evaluate_glove_simon()
    # evaluate_lstm()
    # evaluate_simon_lstm()


def evaluate_simon_lstm():
    size = 100
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset_simon(train_reviews, test_reviews,
                                                             SIMON_MODEL_TRAIN, SIMON_MODEL_TEST)

    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_lstm_model(size, x_train.shape[1])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=16)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_acc.append(scores[1])

    print('SIMON: Accuracy on the training data: {} '.format(np.mean(training_acc)))
    print('SIMON: Accuracy on the test data: {} '.format(np.mean(test_acc)))


def evaluate_lstm():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test, tokenizer = prepare_dataset_lstm(train_reviews, test_reviews)

    training_acc = []
    test_acc = []
    epochs = 16
    for i in range(10):
        model = define_lstm_model(len(tokenizer.word_index) + 1, x_train.shape[1])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 32
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_acc.append(scores[1])

    print('LSTM: Accuracy on the training data: {} '.format(np.mean(training_acc)))
    print('LSTM: Accuracy on the test data: {} '.format(np.mean(test_acc)))


def define_lstm_model(size, input_length):
    embed_dim = 128
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(size, embed_dim, input_length=input_length))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    return model


def evaluate_simon():
    size = 150
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset_simon(train_reviews, test_reviews,
                                                             SIMON_MODEL_TRAIN, SIMON_MODEL_TEST)

    print(x_train.shape)
    training_acc = []
    test_acc = []

    for i in range(10):
        model = define_predicting_model(size, 3)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_acc.append(scores[1])

    print('SIMON: Accuracy on the training data: {} '.format(np.mean(training_acc)))
    print('SIMON: Accuracy on the test data: {} '.format(np.mean(test_acc)))


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

    print('WORD2VEC: Accuracy on the training data: {} '.format(np.mean(training_acc)))
    print('WORD2VEC: Accuracy on the test data: {} '.format(np.mean(test_acc)))


def evaluate_glove_pretrained():
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


def define_predicting_model(input_dim=100, output_dim=2):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=input_dim))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
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

    print('DOC2VEC: Accuracy on the training data: {} '.format(np.mean(training_acc)))
    print('DOC2VEC: Accuracy on the test data: {} '.format(np.mean(test_acc)))


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

    print('DOC2VEC & SIMON: Accuracy on the training data: {} '.format(np.mean(training_acc)))
    print('DOC2VEC & SIMON: Accuracy on the test data: {} '.format(np.mean(test_acc)))


def evaluate_glove_pretrained_simon():
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

    print('WORD2VEC & SIMON: Accuracy on the training data: {}'.format(np.mean(training_acc)))
    print('WORD2VEC & SIMON: Accuracy on the test data: {} '.format(np.mean(test_acc)))


def evaluate_wordvec(three_classes=False):
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    if three_classes:
        word2vec_model = Word2Vec.load(WORD2VEC_MODEL_THREE_CLASSES)
    else:
        word2vec_model = Word2Vec.load(WORD2VEC_MODEL)

    x_train, x_test, y_train, y_test = prepare_dataset_word2vec(word2vec_model, train_reviews, test_reviews, three_classes)

    training_acc = []
    test_acc = []

    for i in range(10):
        if three_classes:
            model = define_predicting_model(150, 3)
        else:
            model = define_predicting_model(100)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=100)
        training_acc.append(history.history['accuracy'][-1])
        scores = model.evaluate(x_test, y_test, verbose=1)
        test_acc.append(scores[1])

    print('WORD2VEC: Accuracy on the training data: {}'.format(np.mean(training_acc)))
    print('WORD2VEC: Accuracy on the test data: {}'.format(np.mean(test_acc)))


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

    print('WORD2VEC & SIMON: Accuracy on the training data: {}'.format(np.mean(training_acc)))
    print('WORD2VEC & SIMON: Accuracy on the test data: {}'.format(np.mean(test_acc)))


def evaluate_glove():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset_glove(GLOVE_VECTORS, train_reviews, test_reviews)

    training_acc = []
    test_acc = []
    dim = 100

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


def evaluate_glove():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset_glove(GLOVE_VECTORS, train_reviews, test_reviews)

    training_acc = []
    test_acc = []
    dim = 100

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


def evaluate_glove_simon():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset_glove(GLOVE_VECTORS, train_reviews, test_reviews)
    x_train2, x_test2, y_train2, y_test2 = prepare_dataset_simon(train_reviews, test_reviews,
                                                             SIMON_MODEL_TRAIN, SIMON_MODEL_TEST)

    x_train_concat = np.zeros((len(x_train), x_train.shape[1] + x_train2.shape[1]))
    x_test_concat = np.zeros((len(x_test), x_test.shape[1] + x_test2.shape[1]))

    training_acc = []
    test_acc = []
    dim = 200

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


if __name__ == "__main__":
    main()
