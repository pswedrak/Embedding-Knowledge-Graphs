import random
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import opinion_lexicon
from sematch.semantic.similarity import WordNetSimilarity
import seaborn as sns
import matplotlib.pyplot as plt


def compute_simon_vector(input_tokens, k=25):
    lexicon_words = extract_lexicon_words(k)
    wns = WordNetSimilarity()
    print(lexicon_words)

    input_tokens_in_wordnet = []
    for input_token in input_tokens:
        if len(wn.synsets(input_token, pos='n')) > 0:
            input_tokens_in_wordnet.append(input_token)
    input_tokens = input_tokens_in_wordnet

    similarity_matrix = np.zeros((len(input_tokens), len(lexicon_words)))
    print(wns.word_similarity('film', 'great', 'wpath'))

    for i in range(len(input_tokens)):
        for j in range(len(lexicon_words)):
            similarity_matrix[i, j] = wns.word_similarity(input_tokens[i], lexicon_words[j], 'wpath')

    sns.heatmap(similarity_matrix, xticklabels=lexicon_words, yticklabels=input_tokens)
    plt.show()

    return np.max(similarity_matrix, axis=0)


def extract_lexicon_words(size):
    positive_words = random.sample(list(opinion_lexicon.positive()), size//2)
    negative_words = random.sample(list(opinion_lexicon.negative()), size - size // 2)

    return positive_words + negative_words
