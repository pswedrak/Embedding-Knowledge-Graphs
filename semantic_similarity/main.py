import numpy as np
import networkx as nx
import gensim.downloader as api
from nltk.corpus import wordnet as wn
from scipy import spatial

from common.constants import SIMILARITY_DATASET, SIMILARITY_RESULT
from semantic_similarity.semantic_pair import SemanticPair


def main():
    wv_from_bin = api.load("glove-twitter-100")
    a = 'cat'
    b = 'cat'
    pairs = load_similarity_dataset(SIMILARITY_DATASET)
    with open(SIMILARITY_RESULT, 'w') as file:
        for pair in pairs:
            a = pair.word_a
            b = pair.word_b
            g, max_depth, rootnode = build_graph(a, b)
            alpha = alpha_coef(a, b, g, max_depth, rootnode)
            sim = compute_similarity(wv_from_bin, a, b)
            file.write(a + " " + b + " " + pair.sim + " " + str(sim * 10))
            file.write('\n')
            file.write(a + " " + b + " " + str(sim * 10 + alpha))
            file.write('\n')
            file.write(b + " " + a + " " + str(sim * 10 - alpha))
            file.write('\n')


def load_similarity_dataset(filename):
    data = []
    with open(filename) as file:
        for line in file.readlines():
            tokens = line.split("	")
            data.append(SemanticPair(tokens[0], tokens[1], tokens[2].split('\n')[0]))
    return data


def alpha_coef(a, b, g, max_depth, rootnode):
    concepts_a = wn.synsets(a, pos='n')
    concepts_b = wn.synsets(b, pos='n')

    dist_sum_a = 0
    for concept in concepts_a:
        dist_sum_a += nx.dijkstra_path_length(g, str(concept.name()), str(rootnode))

    if len(concepts_a) == 0:
        dist_a = 0
    else:
        dist_a = dist_sum_a / len(concepts_a)

    #print(a, dist_a)

    dist_sum_b = 0
    for concept in concepts_b:
        dist_sum_b += nx.dijkstra_path_length(g, str(concept.name()), str(rootnode))

    if len(concepts_b) == 0:
        dist_b = 0
    else:
        dist_b = dist_sum_b / len(concepts_b)

    #print(b, dist_b)

    max_lch_path_length = 0
    for concept_a in concepts_a:
        for concept_b in concepts_b:
            lch_s = concept_a.lowest_common_hypernyms(concept_b)
            # print(concept_a, concept_b)
            for lch in lch_s:
                path_length = nx.dijkstra_path_length(g, str(lch.name()), str(rootnode))
                if path_length > max_lch_path_length:
                    max_lch_path_length = path_length

    #print("lowest common hypernym", max_lch_path_length)
    if max_lch_path_length != 0:
        alpha = (dist_a - dist_b)/max_lch_path_length
    else:
        alpha = 0

    return alpha


def compute_similarity(wv_from_bin, a, b):
    try:
        sim = 1 - spatial.distance.cosine(wv_from_bin[a], wv_from_bin[b])
    except Exception:
        sim = 0
    return sim


def build_graph(word1, word2):
    concepts = wn.synsets(word1, pos='n') + wn.synsets(word2, pos='n')
    if len(concepts) == 0:
        return None, 0
    root = concepts[0].root_hypernyms()[0]

    graph = {root: []}

    for c in concepts:
        graph[c] = []

    for c in concepts:
        insert_hypernyms(graph, c, root)

    for c in concepts:
        for hyponym in c.hyponyms():
            if hyponym not in graph.keys():
                graph[hyponym] = []
            if hyponym not in graph[c]:
                graph[c].append(hyponym)

        for meronym in c.part_meronyms() + c.substance_meronyms():
            if meronym not in graph.keys():
                graph[meronym] = []
            if meronym not in graph[c]:
                graph[c].append(meronym)

        for holonym in c.part_holonyms() + c.substance_holonyms():
            if holonym not in graph.keys():
                graph[holonym] = []
            if holonym not in graph[c]:
                graph[c].append(holonym)

    for c in concepts:
        for cc in c.hyponyms() + c.part_meronyms() + c.substance_meronyms() + c.part_holonyms() + c.substance_holonyms():
            insert_hypernyms(graph, cc, root)

    return build_networx_graph(graph, root)


def build_networx_graph(graph, root):
    g = nx.DiGraph()
    MAX = 20

    for node in graph.keys():
        g.add_node(str(node.name()))

    rootnode = None
    for node in g.nodes():
        if node == root.name():
            rootnode = node

    pos = nx.spring_layout(g)

    for node in g.nodes():
        g.nodes[node]['pos'] = pos[node]

    for node_from in graph.keys():
        for node_to in graph[node_from]:
            g.add_edge(str(node_from.name()), str(node_to.name()), weight=0)
            g.add_edge(str(node_to.name()), str(node_from.name()), weight=0)

    max_depth = 0
    for node in g.nodes():
        depth = nx.shortest_path_length(g, rootnode, node)
        if depth > max_depth:
            max_depth = depth
        g.nodes[node]['depth'] = depth

    for edge in g.edges:
        depth_from = g.nodes()[edge[0]]['depth']
        depth_to = g.nodes()[edge[1]]['depth']
        weight = 1 #- ((depth_from + depth_to)/(2*MAX))
        g.edges[edge]['weight'] = weight

    return g, max_depth, rootnode


def insert_hypernyms(graph, c, root):
    if c == root:
        return
    else:
        for hypernym in c.hypernyms() + c.instance_hypernyms():
            if hypernym in graph.keys():
                if c not in graph[hypernym]:
                    graph[hypernym].append(c)
            else:
                graph[hypernym] = [c]
            insert_hypernyms(graph, hypernym, root)


if __name__ == "__main__":
    main()
