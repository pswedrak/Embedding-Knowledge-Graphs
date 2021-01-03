import random
import jsons

from common.constants import TRAIN_TO_TEST_RATIO
from text_processing.text_utils import clean_line


class Review:
    def __init__(self, stars, text):
        self.stars = stars
        self.text = text
        self.embedding = None

    def __str__(self):
        return "text: " + str(self.text[:50]) + ", stars: " + str(self.stars)


def load_reviews(path, size):
    reviews = []
    with open(path) as f:
        for i in range(size):
            try:
                review = jsons.loads(f.readline())
                reviews.append(Review(review.get('stars'), clean_line(review.get('text'))))
            except UnicodeDecodeError:
                continue
    return reviews


def prepare_data_sets(reviews):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for review in reviews:
        if review.stars < 3:
            if random.random() <= TRAIN_TO_TEST_RATIO:
                train_data.append(review)
                train_labels.append(0)
            else:
                test_data.append(review)
                test_labels.append(0)
        elif review.stars > 3:
            if random.random() <= TRAIN_TO_TEST_RATIO:
                train_data.append(review)
                train_labels.append(1)
            else:
                test_data.append(review)
                test_labels.append(1)

    return train_data, test_data, train_labels, test_labels
