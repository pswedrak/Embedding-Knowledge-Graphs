import networkx as nx
from nltk.corpus import wordnet as wn


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

    return build_networx_graph(graph, root, word1, word2)


def build_networx_graph(graph, root, word1, word2):
    g = nx.DiGraph()
    MAX = 20

    for node in graph.keys():
        g.add_node(str(node.name()))

    rootnode = None
    for node in g.nodes():
        if node == root.name():
            rootnode = node

    pos = nx.random_layout(g)

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

    dist1 = compute_average_distance(g, root, word1)
    dist2 = compute_average_distance(g, root, word2)
    lch_concept, max_lch_path_length = find_lch_concept(g, root, word1, word2)

    return g, max_depth, rootnode, dist1, dist2, lch_concept, max_lch_path_length


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


def compute_average_distance(g, root, target):
    concepts = wn.synsets(target, pos='n')

    dist_sum = 0
    dists = []
    for concept in concepts:
        dist = nx.dijkstra_path_length(g, str(root.name()), str(concept.name()))
        dist_sum += dist
        dists.append(dist)

    if len(concepts) == 0:
        dist = 0
    else:
        #dist = dist_sum / len(concepts)
        dist = min(dists)

    return dist


def find_lch_concept(g, root, word1, word2):
    concepts_a = wn.synsets(word1, pos='n')
    concepts_b = wn.synsets(word2, pos='n')

    max_lch_path_length = 0
    lch_concept = None
    for concept_a in concepts_a:
        for concept_b in concepts_b:
            lch_s = concept_a.lowest_common_hypernyms(concept_b)
            # print(concept_a, concept_b)
            for lch in lch_s:
                if (lch not in concepts_a) & (lch not in concepts_b):
                    path_length = nx.dijkstra_path_length(g, str(root.name()), str(lch.name()))
                    if path_length > max_lch_path_length:
                        max_lch_path_length = path_length
                        lch_concept = lch

    return lch_concept, max_lch_path_length




