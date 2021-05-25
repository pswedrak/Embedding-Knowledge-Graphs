from common.constants import REVIEW_TOKENS_PATH, NEGATIVE_LEXICON, POSITIVE_LEXICON, SIMON_MODEL_TRAIN, \
    SIMON_MODEL_TEST, REVIEW_TEST_TOKENS_PATH, NEUTRAL_LEXICON, SIMON_MODEL_TRAIN_DISSYMMETRY, \
    SIMON_MODEL_TEST_DISSYMETRY
from sentiment_analysis.simon import extract_lexicon_words, build_simon_model, read_lexicon_words
from text_processing.text_utils import process_corpus
from text_processing.yelp_utils import read_reviews


def main():
    reviews = read_reviews(REVIEW_TOKENS_PATH)
    test_reviews = read_reviews(REVIEW_TEST_TOKENS_PATH)
    size = 150

    corpus = process_corpus(reviews)
    dissymetry = True
    # pos_words, neg_words, neu_words, pos_polarity, neg_polarity, neu_polarity = extract_lexicon_words(corpus, size,
    #                                                                                                   POSITIVE_LEXICON,
    #                                                                                                   NEGATIVE_LEXICON,
    #
    #                                                                                                   NEUTRAL_LEXICON)
    (pos_words, pos_polarity), (neg_words, neg_polarity), (neu_words, neu_polarity) = read_lexicon_words(POSITIVE_LEXICON,
                                                                                                   NEGATIVE_LEXICON,
                                                                                                   NEUTRAL_LEXICON)

    print("Lexicon words loaded")

    if dissymetry:
        build_simon_model(reviews + test_reviews, pos_words, neg_words, neu_words, pos_polarity, neg_polarity,
                          neu_polarity, SIMON_MODEL_TRAIN_DISSYMMETRY, SIMON_MODEL_TEST_DISSYMETRY, 1500, dissymetry,
                          True, False)
    else:
        build_simon_model(reviews + test_reviews, pos_words, neg_words, neu_words, pos_polarity, neg_polarity,
                          neu_polarity, SIMON_MODEL_TRAIN, SIMON_MODEL_TEST, 1500, dissymetry, True, False)


if __name__ == "__main__":
    main()
