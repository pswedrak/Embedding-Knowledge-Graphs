import numpy as np
from sematch.semantic.similarity import WordNetSimilarity
from nltk.corpus import sentiwordnet as swn
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical

from common.helpers import store_vectors, store_vector
from semantic_similarity.graph_creator import build_graph
from text_processing.yelp_utils import load_vectors


def build_simon_model(reviews, positive_lexicon_words, negative_lexicon_words, neutral_lexicon_words, pos_polarity,
                      neg_polarity, neu_polarity,
                      train_filename, test_filename, train_size, dissim=True, verbose=True, three_classes=False):
    train_results = []
    test_results = []
    i = 0
    for review in reviews:
        result = list(
            compute_simon_vector(review, positive_lexicon_words, negative_lexicon_words, neutral_lexicon_words,
                                 pos_polarity,
                                 neg_polarity, neu_polarity, dissim, False, False, three_classes=three_classes))
        if i < train_size:
            store_vector(train_filename, result)
        else:
            store_vector(test_filename, result)

        if verbose:
            print("Simon vector has been computed: " + str(i+1) + "/" + str(len(reviews)))

        i += 1

    # store_vectors(train_filename, train_results)
    # store_vectors(test_filename, test_results)


def compute_simon_vector(review, positive_lexicon_words, negative_lexicon_words, neutral_lexicon_words, pos_polarity,
                         neg_polarity, neu_polarity, dissim=True, include_polarity=False, show=False, three_classes=False):
    wns = WordNetSimilarity()
    input_tokens = review.text
    similarity_matrix = np.zeros(
        (len(input_tokens), len(positive_lexicon_words) + len(negative_lexicon_words) + len(neutral_lexicon_words)))
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
            if dissim:
                similarity_matrix[i, j] += (compute_dissim_coef(input_tokens[i], positive_lexicon_words[j][0]) / 10)

        if three_classes:
            for j in range(len(neutral_lexicon_words)):
                similarity_matrix[i, len(positive_lexicon_words) + j] = wns.word_similarity(input_tokens[i],
                                                                                            neutral_lexicon_words[j][0],
                                                                                            'wpath')
                if dissim:
                    similarity_matrix[i, len(positive_lexicon_words) + j] += \
                        (compute_dissim_coef(input_tokens[i], neutral_lexicon_words[j][0]) / 10)

        for j in range(len(negative_lexicon_words)):
            similarity_matrix[i, len(negative_lexicon_words) + len(positive_lexicon_words) + j] = wns.word_similarity(
                input_tokens[i],
                negative_lexicon_words[j][0],
                'wpath')

            if dissim:
                similarity_matrix[i, len(negative_lexicon_words) + len(positive_lexicon_words) + j] += \
                    (compute_dissim_coef(input_tokens[i], negative_lexicon_words[j][0]) / 10)

        print(str(i) + "/" + str(len(input_tokens)))

    if show & three_classes:
        sns.heatmap(similarity_matrix,
                    xticklabels=positive_lexicon_words + neutral_lexicon_words + negative_lexicon_words,
                    yticklabels=input_tokens)
        plt.show()
    elif show & (not three_classes):
        sns.heatmap(similarity_matrix,
                    xticklabels=positive_lexicon_words + negative_lexicon_words,
                    yticklabels=input_tokens)
        plt.show()

    if include_polarity:
        if three_classes:
            return np.multiply(np.max(similarity_matrix, axis=0),  np.concatenate(pos_polarity, neg_polarity, neu_polarity))
        else:
            return np.multiply(np.max(similarity_matrix, axis=0),  np.concatenate(pos_polarity, neg_polarity))
    else:
        return np.max(similarity_matrix, axis=0)


def extract_lexicon_words(corpus, size, pos_file_name, neg_file_name, neu_file_name):
    positive_synsets = {}
    negative_synsets = {}

    pos_words = []
    neg_words = []
    all_neu_words = []
    neu_words = []

    pos_polarity = []
    neg_polarity = []
    neu_polarity = []

    for token in corpus:
        synsets = list(swn.senti_synsets(token))
        if len(synsets) > 0:
            synsets.sort(key=lambda x: x.pos_score(), reverse=True)
            positive_synsets[synsets[0].synset.name()] = synsets[0].pos_score()

            synsets.sort(key=lambda x: x.neg_score(), reverse=True)
            negative_synsets[synsets[0].synset.name()] = synsets[0].neg_score()

            for synset in synsets:
                if (synset.pos_score() == 0) & (synset.neg_score() == 0):
                    all_neu_words.append(synset.synset.name())

    positive_synsets_keys = sorted(positive_synsets, key=positive_synsets.get, reverse=True)

    with open(pos_file_name, "w") as pos_file:
        for i in range(size // 3):
            pos_file.write(positive_synsets_keys[i].split(".")[0] + " " + str(positive_synsets[positive_synsets_keys[i]]))
            pos_file.write('\n')
            pos_words.append(positive_synsets_keys[i].split(".")[0])
            pos_polarity.append(positive_synsets[positive_synsets_keys[i]])

    negative_synsets_keys = sorted(negative_synsets, key=negative_synsets.get, reverse=True)

    with open(neg_file_name, "w") as neg_file:
        for i in range(size // 3):
            neg_file.write(negative_synsets_keys[i].split(".")[0] + " " + str(negative_synsets[negative_synsets_keys[i]]))
            neg_file.write('\n')
            neg_words.append(negative_synsets_keys[i].split(".")[0])
            neg_polarity.append(negative_synsets[negative_synsets_keys[i]])

    with open(neu_file_name, "w") as neu_file:
        i = 0
        j = 0
        while i < (size // 3):
            name = all_neu_words[j].split(".")[0]
            if name not in neu_words:
                neu_file.write(name + " " + "1.0")
                neu_file.write('\n')
                neu_words.append(name)
                neu_polarity.append(1.0)
                i += 1
            j += 1

    return pos_words, neg_words, neu_words, pos_polarity, neg_polarity, neu_polarity


def compute_simon_vectors(reviews, pos_words, neg_words, size, three_classes=False):
    embeddings = []
    for review in reviews:
        simon_vector = compute_simon_vector(review.text, pos_words, neg_words, size, three_classes=three_classes)
        embeddings.append(simon_vector.tolist())
    return embeddings


def prepare_dataset_simon(train_reviews, test_reviews, train_model, test_model, three_classes=False):
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
        elif three_classes & (review.stars == 3):
            x_train.append(simon_vectors[i])
            y_train.append(2)
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
        elif three_classes & (review.stars == 3):
            x_test.append(simon_vectors[i])
            y_test.append(2)
        i += 1

    return np.array(x_train), np.array(x_test), np.array(to_categorical(y_train)), np.array(to_categorical(y_test))


def read_lexicon_words(pos_file_name, neg_file_name, neu_file_name):
    return read_lexicon_file(pos_file_name), read_lexicon_file(neg_file_name), read_lexicon_file(neu_file_name)


def read_lexicon_file(file_name):
    words = []
    polarity = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            tokens = line.split(" ")
            words.append(tokens[0])
            polarity.append(float(tokens[1]))

    return words, polarity


def compute_dissim_coef(word1, word2):
    g, max_depth, root, dist1, dist2, lch_concept, max_lch_path_length = build_graph(word1, word2)
    if max_lch_path_length != 0:
        return (dist1 - dist2) / max_lch_path_length
    else:
        return 0

