import itertools
import jsons

from common.constants import YELP_DATASET_PATH
from sentiment_analysis.simon import compute_simon_vector, extract_lexicon_words
from text_processing.text_utils import process_text_csv, process_corpus_csv, process_directory, process_corpus, \
    load_lexicon_words
import time

from text_processing.yelp_utils import load_reviews, prepare_data_sets
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def main():
    reviews = load_reviews(YELP_DATASET_PATH, 1000)
    train_data, test_data, train_labels, test_labels = prepare_data_sets(reviews)
    documents = [TaggedDocument(doc.text, [i]) for i, doc in enumerate(train_data + test_data)]
    doc2vec_model = Doc2Vec(documents, vector_size=100, window=3, min_count=2)

    for review in train_data + test_data:
        review.embedding = doc2vec_model.infer_vector(review.text)

    with open('doc2vec/output.txt', 'w') as file:
        for review in train_data + test_data:
            file.write(str(review.text))
            file.write('\n')
            file.write(str(review.embedding.tolist()))
            file.write('\n')
            file.write(str(review.stars))
            file.write('\n')


    # size = 20
    # neg_texts = process_directory("texts/reviews/neg")
    # pos_texts = process_directory("texts/reviews/pos")
    # corpus = process_corpus(neg_texts + pos_texts)
    # extract_lexicon_words(corpus, size,
    #                       "texts/reviews/positive_lexicon.txt", "texts/reviews/negative_lexicon.txt")
    # pos_words, polarity = load_lexicon_words("texts/reviews/positive_lexicon.txt")
    # neg_words, polarity = load_lexicon_words("texts/reviews/negative_lexicon.txt")
    # print(compute_simon_vector(neg_texts[0][:30], pos_words, neg_words, size))

    # with open("texts/simons/pos.txt", "w") as file:
    #     for pos_text in pos_texts:
    #         simon_vector = compute_simon_vector(pos_text, pos_words, neg_words, size)
    #         for num in simon_vector:
    #             file.write(str(num) + " ")
    #         file.write('\n')

    # with open("texts/simons/neg.txt", "w") as file:
    #     for neg_text in neg_texts:
    #         simon_vector = compute_simon_vector(neg_text, pos_words, neg_words, size)
    #         for num in simon_vector:
    #             file.write(str(num) + " ")
    #         file.write('\n')

if __name__ == "__main__":
    main()
