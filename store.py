from common.constants import YELP_DATASET_PATH, STORE_REVIEW_TOKENS_PATH
from common.helpers import store_vectors
from doc2vec.doc2vec import generate_doc2vec
from sentiment_analysis.simon import extract_lexicon_words, compute_simon_vectors
from text_processing.text_utils import process_corpus
from text_processing.yelp_utils import load_reviews, store_reviews
from word2vec.word2vec import generate_word2vec


def main():
    reviews = load_reviews(YELP_DATASET_PATH, 1000)
    store_reviews(reviews, STORE_REVIEW_TOKENS_PATH)

    embeddings_doc2vec = generate_doc2vec(reviews)
    store_vectors('results/review_doc2vec.txt', embeddings_doc2vec)

    embeddings_word2vec = generate_word2vec(reviews)
    store_vectors('results/review_word2vec.txt', embeddings_word2vec)

    size = 20
    corpus = process_corpus(reviews)
    pos_words, neg_words, pos_polarity, neg_polarity = extract_lexicon_words(corpus, size,
                                                                             "results/lexicon/positive_lexicon.txt",
                                                                             "results/lexicon/negative_lexicon.txt")

    embeddings_simon = compute_simon_vectors(reviews, pos_words, neg_words, size)
    store_vectors('results/review_simon.txt', embeddings_simon)


if __name__ == "__main__":
    main()
