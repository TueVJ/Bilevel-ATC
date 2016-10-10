import numpy as np
import pandas as pd
import networkx as nx

# Default path
metadata_path = "/media/tue/Data/Github_Dataset/Output_Data/Metadata/"
nodefile = metadata_path + "network_nodes.csv"
linefile = metadata_path + "network_edges.csv"

#5s network
metadata_path = "/home/tue/Dropbox/PhD/Papers/ATCs/Python/5s-data/"
nodefile = metadata_path + "nodes.csv"
linefile = metadata_path + "lines.csv"


def load_network(nodefile=nodefile, linefile=linefile):
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
