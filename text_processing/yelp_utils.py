import random
import jsons

from common.constants import TRAIN_DATA_RATIO
from text_processing.text_utils import clean_line


class Review:
    def __init__(self, stars, text):
        self.stars = stars
        self.text = text
        self.simon_embedding = None
        self.doc2vec_embedding = None
        self.word2vec_embedding = None


def load_reviews_json(path, train_size, test_size):
    train_reviews = []
    test_reviews = []
    with open(path) as f:
        for i in range(train_size):
            try:
                review = jsons.loads(f.readline())
                train_reviews.append(Review(review.get('stars'), clean_line(review.get('text'))))
            except UnicodeDecodeError:
                continue

        for i in range(test_size):
            try:
                review = jsons.loads(f.readline())
                test_reviews.append(Review(review.get('stars'), clean_line(review.get('text'))))
            except UnicodeDecodeError:
                continue

    return train_reviews, test_reviews


def store_reviews(reviews, filename):
    with open(filename, 'w') as file:
        for review in reviews:
            file.write(str(review.text))
            file.write('\n')
            file.write(str(review.stars))
            file.write('\n')


def read_reviews(filename):
    reviews = []
    with open(filename, 'r') as file:
        line = file.readline()
        while len(line) > 0:
            stars = file.readline()
            tokens = list(map(lambda x: x.lstrip()[1:-1], line[1:-2].split(",")))
            review = Review(float(stars[:3]), tokens)
            line = file.readline()
            reviews.append(review)
    return reviews


def prepare_data_sets(reviews):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for review in reviews:
        if review.stars < 3:
            if random.random() <= TRAIN_DATA_RATIO:
                train_data.append(review)
                train_labels.append(0)
            else:
                test_data.append(review)
                test_labels.append(0)
        elif review.stars > 3:
            if random.random() <= TRAIN_DATA_RATIO:
                train_data.append(review)
                train_labels.append(1)
            else:
                test_data.append(review)
                test_labels.append(1)

    return train_data, test_data, train_labels, test_labels


def load_reviews(tokens_file, simon_file, word2vec_file, doc2vec_file):
    reviews = read_reviews(tokens_file)
    simon_vectors = load_vectors(simon_file)
    word2vec = load_vectors(word2vec_file)
    doc2vec = load_vectors(doc2vec_file)

    for i in range(len(reviews)):
        reviews[i].simon_embedding = simon_vectors[i]
        reviews[i].word2vec_embedding = word2vec[i]
        reviews[i].doc2vec_embedding = doc2vec[i]

    return reviews


def load_vectors(filepath):
    vectors = []
    with open(filepath, 'r') as file:
        for line in file.readlines():
            vector = []
            for num in line[1:-2].split(', '):
                vector.append(float(num))
            vectors.append(vector)
    return vectors

