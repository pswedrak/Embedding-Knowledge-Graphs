from common.constants import REVIEW_TOKENS_PATH
from doc2vec.doc2vec import generate_doc2vec_model
from text_processing.yelp_utils import read_reviews


def main():
    reviews = read_reviews(REVIEW_TOKENS_PATH)
    generate_doc2vec_model(reviews)


if __name__ == "__main__":
    main()
