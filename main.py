import itertools

from sentiment_analysis.simon import compute_simon_vector, extract_lexicon_words
from text_processing.text_utils import process_text_csv, process_corpus_csv, process_directory, process_corpus, \
    load_lexicon_words
import time


def main():
    size = 20
    neg_texts = process_directory("texts/reviews/neg")
    pos_texts = process_directory("texts/reviews/pos")
    # corpus = process_corpus(neg_texts + pos_texts)
    # extract_lexicon_words(corpus, size,
    #                       "texts/reviews/positive_lexicon.txt", "texts/reviews/negative_lexicon.txt")
    pos_words, polarity = load_lexicon_words("texts/reviews/positive_lexicon.txt")
    neg_words, polarity = load_lexicon_words("texts/reviews/negative_lexicon.txt")
    print(compute_simon_vector(neg_texts[0], pos_words, neg_words, size))

    # with open("texts/simons/pos.txt", "w") as file:
    #     for pos_text in pos_texts:
    #         simon_vector = compute_simon_vector(pos_text, pos_words, neg_words, size)
    #         for num in simon_vector:
    #             file.write(str(num) + " ")
    #         file.write('\n')

    with open("texts/simons/neg.txt", "w") as file:
        for neg_text in neg_texts:
            simon_vector = compute_simon_vector(neg_text, pos_words, neg_words, size)
            for num in simon_vector:
                file.write(str(num) + " ")
            file.write('\n')


if __name__ == "__main__":
    main()
