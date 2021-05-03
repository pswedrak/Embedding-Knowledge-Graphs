from scipy import spatial

from semantic_similarity.graph_creator import build_graph
from semantic_similarity.graph_drawer import draw_graph
from semantic_similarity.semantic_pair import SemanticPair

import matplotlib.pyplot as plt


def main():
    # wv_from_bin = api.load("glove-twitter-100")
    wv_from_bin = None
    report_similarity('pear', 'fruit', wv_from_bin)
    report_similarity('car', 'vehicle', wv_from_bin)
    report_similarity('phone', 'device', wv_from_bin)


def report_similarity(a, b, wv_from_bin, draw=False):
    g, max_depth, root, dist1, dist2, lch_concept, max_lch_path_length = build_graph(a, b)
    sim = compute_similarity(wv_from_bin, a, b)
    if draw:
        draw_graph(g, a, b, dist1, dist2, lch_concept, max_lch_path_length)
        plt.show()


def load_similarity_dataset(filename):
    data = []
    with open(filename) as file:
        for line in file.readlines():
            tokens = line.split("	")
            data.append(SemanticPair(tokens[0], tokens[1], tokens[2].split('\n')[0]))
    return data


def compute_similarity(wv_from_bin, a, b):
    try:
        sim = 1 - spatial.distance.cosine(wv_from_bin[a], wv_from_bin[b])
    except Exception:
        sim = 0
    return sim


if __name__ == "__main__":
    main()
