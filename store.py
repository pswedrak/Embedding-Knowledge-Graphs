from common.constants import YELP_DATASET_PATH, REVIEW_TOKENS_PATH, REVIEW_TEST_TOKENS_PATH
from text_processing.yelp_utils import store_reviews, load_reviews_json


def main():
    test_reviews, train_reviews = load_reviews_json(YELP_DATASET_PATH, 700, 300)
    store_reviews(test_reviews, REVIEW_TOKENS_PATH)
    store_reviews(train_reviews, REVIEW_TEST_TOKENS_PATH)


if __name__ == "__main__":
    main()
