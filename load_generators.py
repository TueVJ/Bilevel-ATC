import numpy as np
import pandas as pd
import networkx as nx


def load_generators(generatorfile='6bus-data/generators.csv'):
    generators = pd.read_csv(generatorfile)
    return generators.set_index('ID')
