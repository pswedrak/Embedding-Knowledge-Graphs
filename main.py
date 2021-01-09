from gensim.models import Doc2Vec
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from common.constants import DOC2VEC_MODEL, REVIEW_TOKENS_PATH, REVIEW_TEST_TOKENS_PATH, SIMON_MODEL_TRAIN, \
    SIMON_MODEL_TEST
from doc2vec.doc2vec import prepare_dataset_doc2vec
from sentiment_analysis.simon import prepare_dataset_simon
from text_processing.yelp_utils import read_reviews
from word2vec.word2vec import prepare_dataset_word2vec


def main():
    # evaluate_doc2vec()
    # evaluate_word2vec()
    evaluate_simon()


def evaluate_simon():
    size = 50
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset_simon(train_reviews, test_reviews,
                                                             SIMON_MODEL_TRAIN, SIMON_MODEL_TEST)

    model = define_predicting_model(size)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print('SIMON: Accuracy on test data: {}% \n Error on test data: {}'.format(scores[1], 1 - scores[1]))


def evaluate_word2vec():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    x_train, x_test, y_train, y_test = prepare_dataset_word2vec(train_reviews, test_reviews)

    model = define_predicting_model()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print('WORD2VEC: Accuracy on test data: {}% \n Error on test data: {}'.format(scores[1], 1 - scores[1]))


def define_predicting_model(input_dim=100):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=input_dim))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


def evaluate_doc2vec():
    train_reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    doc2vec_model = Doc2Vec.load(DOC2VEC_MODEL)
    x_train, x_test, y_train, y_test = prepare_dataset_doc2vec(doc2vec_model, train_reviews, test_reviews)

    model = define_predicting_model()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print('DOC2VEC: Accuracy on test data: {}% \n Error on test data: {}'.format(scores[1], 1 - scores[1]))


if __name__ == "__main__":
    main()
