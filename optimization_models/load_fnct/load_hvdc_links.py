import numpy as np
import pandas as pd
import networkx as nx

# Default path
metadata_path = "/media/tue/Data/Github_Dataset/Output_Data/Metadata/"
dcfile = metadata_path + "network_hvdc_links.csv"


def load_hvdc_links(dcfile=dcfile):
    dclinks = pd.read_csv(dcfile).set_index('ID')

    return dclinks
