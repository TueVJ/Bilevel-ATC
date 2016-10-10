# coding=UTF8
import pandas as pd
import numpy as np
import networkx as nx
import gurobipy as gb
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import izip, cycle
from collections import defaultdict

from optimization_models import DA_Zonal, DA_Nodal, RT_Nodal, Smeers_ATC_Maximization_nodes


gamma = 0.5
alpha = 0.75
loadhour = 8
season = 'Summer'

# Load full 5s network
loaddir = '5s-data/'

stochstore = pd.HDFStore('optimalstore_stoch.h5')
optimal_cap_df = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_cap']
bounddf = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_lower_bound']
windrt = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/windrt']
solarrt = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/solarrt']
windfc = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/windfc']
solarfc = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/solarfc']
loadts = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/loadts']
stochstore.close()

print "Data loaded"

zda = DA_Zonal(gamma*alpha*windfc, gamma*(1-alpha)*solarfc, loadts)
nda = DA_Nodal(gamma*alpha*windfc, gamma*(1-alpha)*solarfc, loadts)
nrts = [RT_Nodal(gamma*alpha*windts, gamma*(1-alpha)*solarrt[s], loadts) for s, windts in windrt.iteritems()]

print "Models Built"

zda.model.setParam('OutputFlag', False)
nda.model.setParam('OutputFlag', False)
[nrt.model.setParam('OutputFlag', False) for nrt in nrts]

####
# Nodal DA
####

print "Optimize Nodal"

nda.optimize()
for nrt in nrts:
    nrt._update_DA_schedule(nda)
    nrt.optimize()

DA_Heuristic = []
RT_Heuristic = []
Labels_Heuristic = []
DA_Heuristic.append(nda.model.getObjective().getValue())
RT_Heuristic.append(sum(nrt.model.getObjective().getValue() for nrt in nrts)/len(nrts))
Labels_Heuristic.append('Nodal DA')

print "Nodal Done"

###
#   Smeers' heuristic, from
#   G. Oggioni, Y. Smeers: "Market failures of Market Coupling
#    and counter-trading in Europe:
#    An illustrative model based discussion"
###

print "Load Smeers"

store = pd.HDFStore('heuristicstore.h5')
try:
    smeers_df = store['layouts/smeers']
    smeers_cap_dict = {tuple(k): v for k, v in smeers_df.to_dict()[0].iteritems()}
    print "Smeers loaded"
except KeyError:
    print "Smeers not found: Regeneration started"
    smeers = Smeers_ATC_Maximization_nodes(nrt.data.nodeorder[0], nrt.data.nodeorder[1])
    smeers_cap_dict = {}
    smeers_minimizing_nodes = {}
    smeers.model.setParam('OutputFlag', False)
    for i, e in enumerate(zda.data.zoneedgeorder):
        print '{0}, {1:.01f}\%'.format(e, i*100.0/len(zda.data.zoneedgeorder))
        mincap = 1e50
        z1 = zda.data.zone_to_nodes[e[0]]
        z2 = zda.data.zone_to_nodes[e[1]]
        for n1 in z1:
            for n2 in z2:
                smeers.update_injection_nodes(n1, n2)
                smeers.optimize()
                # If the result was unbounded or infeasible, ignore the solution.
                if smeers.model.status == gb.GRB.OPTIMAL:
                    ob = smeers.model.getObjective().getValue()
                    if ob == 0:
                        print 'zero objective'
                        raise SystemExit
                    if ob < mincap:
                        smeers_minimizing_nodes[e] = (n1, n2)
                        mincap = ob
        smeers_cap_dict[e] = mincap

    smeers_df = pd.DataFrame(
        smeers_cap_dict.values(),
        index=pd.MultiIndex.from_tuples(
            smeers_cap_dict.keys(),
            names=['fromNode', 'toNode'])
    )

    store['layouts/smeers'] = smeers_df
store.close()

# obj_smeers_zonal = []
# obj_smeers_nodal = []
# for alpha in alphas:
#     for e, c in smeers_cap_dict.iteritems():
#         for t in zda.data.taus:
#             zda.variables.edgeflow[e, t].ub = c*alpha
#             zda.variables.edgeflow[e, t].lb = -c*alpha
#     zda.variables.edgeflow[('ITA', 'GRC'), t].ub = 0.0
#     zda.variables.edgeflow[('ITA', 'GRC'), t].lb = 0.0
#     zda.optimize()
#     nrt._update_DA_schedule(zda)
#     nrt.optimize()
#     obj_smeers_zonal.append(zda.model.getObjective().getValue())
#     obj_smeers_nodal.append(nrt.model.getObjective().getValue())

# smeersdf = pd.DataFrame({
#     'DA': obj_smeers_zonal,
#     'RT': obj_smeers_nodal},
#     index=alphas)

# store.open()
# store['results/smeers/'+str(gamma)] = smeersdf
# store.close()

# Smeers with no scaling
for e, c in smeers_cap_dict.iteritems():
    for t in zda.data.taus:
        zda.variables.edgeflow[e, t].ub = c*alpha
        zda.variables.edgeflow[e, t].lb = -c*alpha
zda.variables.edgeflow[('ITA', 'GRC'), t].ub = 0.0
zda.variables.edgeflow[('ITA', 'GRC'), t].lb = 0.0
zda.optimize()
for nrt in nrts:
    nrt._update_DA_schedule(zda)
    nrt.optimize()

DA_Heuristic.append(zda.model.getObjective().getValue())
RT_Heuristic.append(sum(nrt.model.getObjective().getValue() for nrt in nrts)/len(nrts))
Labels_Heuristic.append('Min/Max flow')

print "Smeers done"

###
#  Load 'optimal' layout
###

print "Loading optimized layout"

store = pd.HDFStore('optimalstore_stoch.h5')
optimal_cap_df = store[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_cap']
bounds = store[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_lower_bound']
max_lower_bound = bounds['lower_bounds'].iloc[-1]
store.close()
optimal_cap_dict = {tuple(k): v for k, v in optimal_cap_df.iloc[:, 0].to_dict().iteritems()}

for e, c in optimal_cap_dict.iteritems():
    for t in zda.data.taus:
        try:
            zda.variables.edgeflow[e, t].ub = c*alpha
            zda.variables.edgeflow[e, t].lb = -c*alpha
        except KeyError:
            zda.variables.edgeflow[e[::-1], t].ub = c*alpha
            zda.variables.edgeflow[e[::-1], t].lb = -c*alpha
zda.variables.edgeflow[('ITA', 'GRC'), t].ub = 0.0
zda.variables.edgeflow[('ITA', 'GRC'), t].lb = 0.0
zda.optimize()
for nrt in nrts:
    nrt._update_DA_schedule(zda)
    nrt.optimize()

DA_Heuristic.append(zda.model.getObjective().getValue())
RT_Heuristic.append(sum(nrt.model.ObjVal for nrt in nrts)/len(nrts))
Labels_Heuristic.append('Optimized\n(1% gap)')


resdf = pd.DataFrame({
    'Season': season,
    'Renewable Penetration': gamma,
    'ProcessType': Labels_Heuristic,
    'DA Cost': DA_Heuristic,
    'RT Cost': RT_Heuristic,
    'Hour': loadhour})

outstore = pd.HDFStore('outputstore_stoch.h5')

raise SystemExit

try:
    outstore['results'] = pd.concat((outstore['results'], resdf), ignore_index=True)
except KeyError:
    outstore['results'] = resdf
outstore.close()

raise SystemExit
###
# Plotting
###

# plt.figure()
# ax = plt.axes()
# c = sns.xkcd_rgb['deep red']
# plt.plot(scaledf.index, scaledf['DA'], ls='--', c=c)
# plt.plot(scaledf.index, scaledf['RT'], c=c, label='Scaled underlying capacities')

# c = sns.xkcd_rgb['deep blue']
# plt.plot(smeersdf.index, smeersdf['DA'], ls='--', c=c)
# plt.plot(smeersdf.index, smeersdf['RT'], c=c, label='Scaled Smeers\' Heuristic')

# c = sns.xkcd_rgb['charcoal']
# i_nodal = Labels_Heuristic.index('Nodal DA')
# plt.axhline(DA_Heuristic[i_nodal], ls='--', c=c)
# plt.axhline(RT_Heuristic[i_nodal], c=c, label='Nodal DA')

colorcycle = sns.color_palette()

plt.figure(figsize=(6, 4))
ax2 = plt.axes()
heurdf = pd.DataFrame(index=['DA Cost', 'RT Cost'], data={l: [da, rt] for l, da, rt in zip(Labels_Heuristic, DA_Heuristic, RT_Heuristic)}).T
(heurdf/1000).ix[['Nodal DA', 'Optimized\n(1% gap)', 'Min/Max flow']].plot(kind='bar', ax=ax2, color=colorcycle)
rects = ax2.patches

for i, rect in enumerate(rects):
    h = rect.get_height()
    label = u'{:.00f}'.format(round(h))
    if rect.get_facecolor()[1] > rect.get_facecolor()[2]:
        h = h*1.05
    ax2.text(rect.get_x() + rect.get_width()/2, h + 10, label, ha='center', va='bottom')

c = sns.xkcd_rgb['charcoal']
plt.axhline(max_lower_bound/1000, ls='--', c=c, alpha=0.7)

plt.xticks(rotation=0)
plt.ylabel(u'Final cost [k$]')
plt.tight_layout()
plt.legend(loc='upper left')
sns.despine()
plt.draw()
plt.ylim([0.9*heurdf.min().min()/1000, 1.10*heurdf.max().max()/1000])
plt.savefig('../pic/heuristic_comparison_stoch.pdf')

plt.figure(figsize=(4, 3))
ax3 = plt.axes()
c = sns.xkcd_rgb['deep red']
plt.plot(bounds['lower_bounds']/1000, c=c, lw=2, marker='.')
c = sns.xkcd_rgb['charcoal']
plt.plot(bounds['upper_bounds']/1000, c=c, lw=2, marker='.')
plt.ylabel(u'Final cost [k$]')
plt.xlabel(u'Benders\' cut')
sns.despine()
plt.tight_layout()
plt.axvline(25, ls='--', c='k', alpha=0.6)
plt.savefig('../pic/benders_bounds_stoch.pdf')

raise SystemExit

import networkx as nx
from mpl_toolkits.basemap import Basemap
from myhelpers import symmetrize_dict
mymap = Basemap(-10, 35, 30, 60, resolution='i')

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
# plt.savefig('../pic/zonal_link_map_blue_is_zero.pdf')

raise SystemExit

###
#  Calculate optimal RT dispatch
###

for k in nrt.variables.gprod_DA.iterkeys():
    nrt.variables.gprod_DA[k].ub = nrt.variables.gprod[k].ub
    nrt.variables.gprod_DA[k].lb = nrt.variables.gprod[k].lb

nrt.optimize()
