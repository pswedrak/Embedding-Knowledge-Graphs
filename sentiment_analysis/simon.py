import numpy as np
from sematch.semantic.similarity import WordNetSimilarity
from nltk.corpus import sentiwordnet as swn
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical

from common.helpers import store_vectors
from text_processing.yelp_utils import load_vectors


def build_simon_model(reviews, positive_lexicon_words, negative_lexicon_words, train_filename, test_filename,
                      train_size, verbose=True):
    train_results = []
    test_results = []
    i = 0
    for review in reviews:
        if i < train_size:
            train_results.append(list(compute_simon_vector(review, positive_lexicon_words, negative_lexicon_words)))
        else:
            test_results.append(list(compute_simon_vector(review, positive_lexicon_words, negative_lexicon_words)))

        if verbose:
            print("Simon vector has been computed: " + str(i+1) + "/" + str(len(reviews)))

        i += 1

    store_vectors(train_filename, train_results)
    store_vectors(test_filename, test_results)


def compute_simon_vector(review, positive_lexicon_words, negative_lexicon_words, show=False):
    wns = WordNetSimilarity()
    input_tokens = review.text
    similarity_matrix = np.zeros((len(input_tokens), len(positive_lexicon_words) + len(negative_lexicon_words)))
    # input_tokens_in_wordnet = []
    # for input_token in input_tokens:
    #     if len(wn.synsets(input_token, pos='n')) > 0:
    #         input_tokens_in_wordnet.append(input_token)
    # input_tokens = input_tokens_in_wordnet
    #
    # similarity_matrix = np.zeros((len(input_tokens), len(lexicon_words)))
    # print(wns.word_similarity('film', 'great', 'wpath'))
    #
    for i in range(len(input_tokens)):
            for j in range(len(positive_lexicon_words)):
                similarity_matrix[i, j] = wns.word_similarity(input_tokens[i], positive_lexicon_words[j][0], 'wpath')

            for j in range(len(negative_lexicon_words)):
                similarity_matrix[i, len(positive_lexicon_words) + j] = wns.word_similarity(input_tokens[i], negative_lexicon_words[j][0], 'wpath')

    if show:
        sns.heatmap(similarity_matrix, xticklabels=positive_lexicon_words+negative_lexicon_words, yticklabels=input_tokens)
        plt.show()

    return np.max(similarity_matrix, axis=0)


def extract_lexicon_words(corpus, size, pos_file_name, neg_file_name):
    positive_synsets = {}
    negative_synsets = {}

    pos_words = []
    neg_words = []
    pos_polarity = []
    neg_polarity = []

    for token in corpus:
        synsets = list(swn.senti_synsets(token))
        if len(synsets) > 0:
            synsets.sort(key=lambda x: x.pos_score(), reverse=True)
            positive_synsets[synsets[0].synset.name()] = synsets[0].pos_score()

            synsets.sort(key=lambda x: x.neg_score(), reverse=True)
            negative_synsets[synsets[0].synset.name()] = synsets[0].neg_score()

    positive_synsets_keys = sorted(positive_synsets, key=positive_synsets.get, reverse=True)

    with open(pos_file_name, "w") as pos_file:
        for i in range(size//2):
            pos_file.write(positive_synsets_keys[i].split(".")[0] + " " + str(positive_synsets[positive_synsets_keys[i]]))
            pos_file.write('\n')
            pos_words.append(positive_synsets_keys[i].split(".")[0])
            pos_polarity.append(positive_synsets[positive_synsets_keys[i]])

    negative_synsets_keys = sorted(negative_synsets, key=negative_synsets.get, reverse=True)

    with open(neg_file_name, "w") as neg_file:
        for i in range(size - size // 2):
            neg_file.write(negative_synsets_keys[i].split(".")[0] + " " + str(negative_synsets[negative_synsets_keys[i]]))
            neg_file.write('\n')
            neg_words.append(negative_synsets_keys[i].split(".")[0])
            neg_polarity.append(negative_synsets[negative_synsets_keys[i]])

    return pos_words, neg_words, pos_polarity, neg_polarity


def compute_simon_vectors(reviews, pos_words, neg_words, size):
    embeddings = []
    for review in reviews:
        simon_vector = compute_simon_vector(review.text, pos_words, neg_words, size)
        embeddings.append(simon_vector.tolist())
    return embeddings


def prepare_dataset_simon(train_reviews, test_reviews, train_model, test_model):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    simon_vectors = load_vectors(train_model)
    i = 0
    for review in train_reviews:
        if review.stars <= 2:
            x_train.append(simon_vectors[i])
            y_train.append(0)
        elif review.stars >= 4:
            x_train.append(simon_vectors[i])
            y_train.append(1)
        i += 1

    simon_vectors = load_vectors(test_model)
    i = 0
    for review in test_reviews:
        if review.stars <= 2:
            x_test.append(simon_vectors[i])
            y_test.append(0)
        elif review.stars >= 4:
            x_test.append(simon_vectors[i])
            y_test.append(1)
        i += 1

    return np.array(x_train), np.array(x_test), np.array(to_categorical(y_train)), np.array(to_categorical(y_test))



