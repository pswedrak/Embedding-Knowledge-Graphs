from common.constants import REVIEW_TOKENS_PATH, NEGATIVE_LEXICON, POSITIVE_LEXICON, SIMON_MODEL_TRAIN, \
    SIMON_MODEL_TEST, REVIEW_TEST_TOKENS_PATH, NEUTRAL_LEXICON
from sentiment_analysis.simon import extract_lexicon_words, build_simon_model
from text_processing.text_utils import process_corpus
from text_processing.yelp_utils import read_reviews


def main():
    reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    size = 150

    corpus = process_corpus(reviews)
    pos_words, neg_words, neu_words, pos_polarity, neg_polarity, neu_polarity = extract_lexicon_words(corpus, size, POSITIVE_LEXICON, NEGATIVE_LEXICON, NEUTRAL_LEXICON)
    build_simon_model(reviews + test_reviews, pos_words, neg_words, pos_polarity, neg_polarity, SIMON_MODEL_TRAIN, SIMON_MODEL_TEST, 1500)


if __name__ == "__main__":
    main()
