import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from myhelpers import symmetrize_dict
import pandas as pd

from optimization_models import DA_Zonal

sns.set_style('ticks')

# Load full 5s network
gamma = 0.5
alpha = 0.75
loadhour = 18
season = 'Summer'

# Load full 5s network
loaddir = '5s-data/'

wind_da = pd.read_csv(loaddir + season + '/wind_fc.csv', index_col='Time').ix[[loadhour]]
loadts = pd.read_csv(loaddir + season + '/load.csv', index_col='Time').ix[[loadhour]]

for df in [wind_da, loadts]:
    df.columns = map(int, df.columns)

zda = DA_Zonal(wind_da, wind_da, loadts)

mymap = Basemap(-10, 36, 26, 58, resolution='i')

store = pd.HDFStore('optimalstore_stoch.h5')
optimal_cap_df = store[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_cap']
bounds = store[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_lower_bound']
max_lower_bound = bounds['lower_bounds'].iloc[-1]
store.close()
optimal_cap_dict = {tuple(k): v for k, v in optimal_cap_df.iloc[:, 0].iteritems()}

plt.figure(figsize=(6, 4))
zG = nx.from_edgelist(zda.data.zoneedgeorder)
pos = nx.get_node_attributes(zda.data.G, 'pos')
nx.set_node_attributes(zG, 'pos', {z: np.mean([pos[n] for n in zda.data.zone_to_nodes[z]], axis=0) for z in zda.data.zoneorder})
zpos = nx.get_node_attributes(zG, 'pos')
thecaps = optimal_cap_dict.copy()
thecaps['ITA', 'GRC'] = 0
symoptdict = symmetrize_dict(thecaps)
mymap.drawcoastlines(color=sns.xkcd_rgb['charcoal'])
mymap.fillcontinents(zorder=-100)

nx.draw_networkx_nodes(zG, pos=zpos, node_size=40, node_color='k')
nx.draw_networkx_edges(
    zG, pos=zpos, width=3,
    edge_color=[int(symoptdict[e] > 0.01) for e in zda.data.zoneedgeorder],
    edgelist=zda.data.zoneedgeorder,
    edge_cmap=sns.blend_palette(sns.xkcd_palette(['magenta', 'grass green']), as_cmap=True))
l1 = plt.Line2D([], [], ls='-', lw=3, c=sns.xkcd_rgb['magenta'])
l2 = plt.Line2D([], [], ls='-', lw=3, c=sns.xkcd_rgb['grass green'])
plt.legend([l1, l2], ['ATC = 0', 'ATC > 0'], loc='upper right', ncol=1, frameon=True, fontsize=14)
plt.tight_layout()
plt.savefig('../pic/zonal_link_map_hour_' + str(loadhour) + '.pdf')
