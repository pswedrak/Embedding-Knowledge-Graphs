from common.constants import REVIEW_TOKENS_PATH, NEGATIVE_LEXICON, POSITIVE_LEXICON, SIMON_MODEL_TRAIN, \
    SIMON_MODEL_TEST, REVIEW_TEST_TOKENS_PATH
from sentiment_analysis.simon import extract_lexicon_words, build_simon_model
from text_processing.text_utils import process_corpus
from text_processing.yelp_utils import read_reviews


def main():
    reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    size = 50
    corpus = process_corpus(reviews)
    pos_words, neg_words, _, _ = extract_lexicon_words(corpus, size, POSITIVE_LEXICON, NEGATIVE_LEXICON)
    build_simon_model(reviews + test_reviews, pos_words, neg_words, SIMON_MODEL_TRAIN, SIMON_MODEL_TEST, 700)


if __name__ == "__main__":
    main()
