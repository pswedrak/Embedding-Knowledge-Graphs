from common.constants import STORE_REVIEW_TOKENS_PATH, SIMON_FILE, WORD2VEC_FILE, DOC2VEC_FILE
from text_processing.yelp_utils import load_reviews


def main():
    reviews = load_reviews(STORE_REVIEW_TOKENS_PATH, SIMON_FILE, WORD2VEC_FILE, DOC2VEC_FILE)
    print(reviews[0].text)
    print(reviews[0].stars)
    print(reviews[0].simon_embedding)
    print(reviews[0].word2vec_embedding)
    print(reviews[0].doc2vec_embedding)


if __name__ == "__main__":
    main()
