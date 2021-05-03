import plotly.graph_objects as go
from nltk.corpus import wordnet as wn


def draw_graph(G, word1, word2, dist1, dist2, lch_concept, max_lch_path_length):
    alpha = (dist1 - dist2) / max_lch_path_length
    fig = go.Figure(layout=go.Layout(
                        title='<br>Graph created for words: ' + word1 + ' and ' + word2,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Average distance from root to " + word1 + ": " + str(dist1)
                                 + " Average distance from root to " + word2 + ": " + str(dist2)
                                 + " Average distance from root to " + str(lch_concept.name()) + ": " + str(max_lch_path_length)
                                 + " Computed coefficient: " + str(alpha),
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    concepts1 = wn.synsets(word1, pos='n')
    concepts1 = list(map(lambda concept: concept.name(), concepts1))

    concepts2 = wn.synsets(word2, pos='n')
    concepts2 = list(map(lambda concept: concept.name(), concepts2))

    for edge in G.edges():
        edge_x = []
        edge_y = []
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_y.append(y0)
        edge_x.append(x1)
        edge_y.append(y1)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='lightblue'),
            hoverinfo='text',
            textposition='top right',
            mode='lines')
        fig.add_trace(edge_trace)

    node_x = []
    node_y = []
    node_colours = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        if node in concepts1:
            node_colours.append('#ff0000')
        elif node in concepts2:
            node_colours.append('#000000')
        else:
            node_colours.append('#ffffff')

    node_text = []

    for node in G.nodes.items():
        node_text.append(node[0] + ' depth: ' + str(node[1]['depth']))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colours,
            size=10,
            line_width=2))
    fig.add_trace(node_trace)
    fig.show()
