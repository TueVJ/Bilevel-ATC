import numpy as np
import pandas as pd
import networkx as nx


def load_network(nodefile='6bus-data/nodes.csv', linefile='6bus-data/lines.csv'):
    nodes = pd.read_csv(nodefile)
    lines = pd.read_csv(linefile)

    G = nx.Graph()
    G.add_nodes_from(nodes.ID.values)
    G.add_edges_from(lines.set_index(['fromNode', 'toNode']).index)

    nodes['pos'] = [(lon, lat) for lon, lat in zip(nodes.longitude, nodes.latitude)]

    for cn, s in nodes.set_index('ID').iteritems():
        nx.set_node_attributes(G, cn, s.to_dict())

    for cn, s in lines.set_index(['fromNode', 'toNode']).iteritems():
        nx.set_edge_attributes(G, cn, s.to_dict())

    return G
