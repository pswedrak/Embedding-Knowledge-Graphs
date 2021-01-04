import itertools
import jsons

from common.constants import YELP_DATASET_PATH, STORE_REVIEW_TOKENS_PATH
from sentiment_analysis.simon import compute_simon_vector, extract_lexicon_words
from text_processing.text_utils import process_text_csv, process_corpus_csv, process_directory, process_corpus, \
    load_lexicon_words
import time

from text_processing.yelp_utils import load_reviews, prepare_data_sets, store_reviews, read_reviews
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from word2vec.word2vec import generate_word2vec


def main():
    # reviews = load_reviews(YELP_DATASET_PATH, 1000)
    # store_reviews(reviews, STORE_REVIEW_TOKENS_PATH)
    reviews = read_reviews(STORE_REVIEW_TOKENS_PATH)
    train_data, test_data, train_labels, test_labels = prepare_data_sets(reviews)
    # documents = [TaggedDocument(doc.text, [i]) for i, doc in enumerate(train_data + test_data)]
    # doc2vec_model = Doc2Vec(documents, vector_size=100, window=3, min_count=2)
    #
    # for review in train_data + test_data:
    #     review.embedding = doc2vec_model.infer_vector(review.text)
    #
    # with open('results/review_doc2vec.txt', 'w') as file:
    #     for review in train_data + test_data:
    #         file.write(str(review.embedding.tolist()))
    #         file.write('\n')

    # generate_word2vec(train_data)

    size = 20
    # neg_texts = process_directory("results/reviews/neg")
    # pos_texts = process_directory("results/reviews/pos")
    # corpus = process_corpus(reviews)
    # print(corpus)
    # extract_lexicon_words(corpus, size,
    #                       "results/lexicon/positive_lexicon.txt", "results/lexicon/negative_lexicon.txt")
    pos_words, polarity = load_lexicon_words("results/lexicon/positive_lexicon.txt")
    neg_words, polarity = load_lexicon_words("results/lexicon/negative_lexicon.txt")
    # print(compute_simon_vector(neg_texts[0][:30], pos_words, neg_words, size))

    with open("results/review_simon.txt", "w") as file:
        for review in reviews:
            print(review)
            simon_vector = compute_simon_vector(review.text, pos_words, neg_words, size)
            file.write(str(simon_vector.tolist()))
            file.write('\n')

    # with open("results/simons/neg.txt", "w") as file:
    #     for neg_text in neg_texts:
    #         simon_vector = compute_simon_vector(neg_text, pos_words, neg_words, size)
    #         for num in simon_vector:
    #             file.write(str(num) + " ")
    #         file.write('\n')


if __name__ == "__main__":
    main()
