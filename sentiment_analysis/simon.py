import random
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import opinion_lexicon


def compute_simon_vector(input_tokens, k=25):
    lexicon_words = extract_lexicon_words(k)
    similarity_matrix = np.zeros((len(input_tokens), len(lexicon_words)))

    input_tokens_in_wordnet = []
    for input_token in input_tokens:
        if len(wn.synsets(input_token, pos='n')) > 0:
            input_tokens_in_wordnet.append(input_token)
    input_tokens = input_tokens_in_wordnet

    for i in range(len(input_tokens)):
        for j in range(len(lexicon_words)):
            similarity_matrix[i, j] = 1

    return np.max(similarity_matrix, axis=0)


def extract_lexicon_words(size):
    positive_words = random.sample(list(opinion_lexicon.positive()), size//2)
    negative_words = random.sample(list(opinion_lexicon.positive()), size - size // 2)

    return positive_words + negative_words
