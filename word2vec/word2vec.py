import numpy as np
from gensim.models import Word2Vec


def generate_word2vec(reviews):
    sentences = list(map(lambda x: x.text, reviews))
    model = Word2Vec(
        sentences,
        size=100,
        window=5,
        min_count=1,
        workers=10,
        iter=500)

    results = []
    for sentence in sentences:
        vectors = np.array(list(map(lambda x: model.wv[x], sentence)))
        results.append(np.mean(vectors, axis=0).tolist())

    return results
