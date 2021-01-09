import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from tensorflow.python.keras.utils.np_utils import to_categorical

from common.constants import DOC2VEC_MODEL


def generate_doc2vec_model(reviews):
    documents = [TaggedDocument(doc.text, [i]) for i, doc in enumerate(reviews)]
    doc2vec_model = Doc2Vec(documents, vector_size=100, window=3, min_count=1)
    doc2vec_model.save(DOC2VEC_MODEL)

    return doc2vec_model


def prepare_dataset_doc2vec(model, train_reviews, test_reviews):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for review in train_reviews:
        if review.stars <= 2:
            x_train.append(model.infer_vector(review.text))
            y_train.append(0)
        elif review.stars >= 4:
            x_train.append(model.infer_vector(review.text))
            y_train.append(1)

    for review in test_reviews:
        if review.stars <= 2:
            x_test.append(model.infer_vector(review.text))
            y_test.append(0)
        elif review.stars >= 4:
            x_test.append(model.infer_vector(review.text))
            y_test.append(1)

    return np.array(x_train), np.array(x_test), np.array(to_categorical(y_train)), np.array(to_categorical(y_test))
