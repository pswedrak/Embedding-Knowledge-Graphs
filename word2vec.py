from common.constants import REVIEW_TOKENS_PATH
from text_processing.yelp_utils import read_reviews
from word2vec.word2vec import generate_word2vec_model


def main():
    reviews = read_reviews(REVIEW_TOKENS_PATH)
    generate_word2vec_model(reviews, True)


if __name__ == "__main__":
    main()
