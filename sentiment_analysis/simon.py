from collections import Counter

import numpy as np
from nltk.corpus import wordnet as wn
from sematch.semantic.similarity import WordNetSimilarity
from nltk.corpus import sentiwordnet as swn
import seaborn as sns
import matplotlib.pyplot as plt


def compute_simon_vector(input_tokens, positive_lexicon_words, negative_lexicon_words, size=25):
    wns = WordNetSimilarity()
    similarity_matrix = np.zeros((len(input_tokens), size))
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

    #sns.heatmap(similarity_matrix, xticklabels=positive_lexicon_words+negative_lexicon_words, yticklabels=input_tokens)
    #plt.show()

    return np.max(similarity_matrix, axis=0)


def extract_lexicon_words(corpus, size, pos_file_name, neg_file_name):
    positive_synsets = {}
    negative_synsets = {}

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

    negative_synsets_keys = sorted(negative_synsets, key=negative_synsets.get, reverse=True)

    with open(neg_file_name, "w") as neg_file:
        for i in range(size - size // 2):
            neg_file.write(negative_synsets_keys[i].split(".")[0] + " " + str(negative_synsets[negative_synsets_keys[i]]))
            neg_file.write('\n')



