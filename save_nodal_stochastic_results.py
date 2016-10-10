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
loadhours = range(24)
season = 'Summer'

# Load full 5s network
loaddir = '5s-data/'

directdispatch = pd.HDFStore('stoch_nodal_dispatch.h5')['direct_nodal']

stochstore = pd.HDFStore('optimalstore_stoch.h5')
optimal_cap_df = stochstore[season + '/' + str(loadhours[0]) + '/' + str(gamma) + '/optimized_cap']
bounddf = stochstore[season + '/' + str(loadhours[0]) + '/' + str(gamma) + '/optimized_lower_bound']
windrt = stochstore[season + '/' + str(loadhours[0]) + '/' + str(gamma) + '/windrt']
solarrt = stochstore[season + '/' + str(loadhours[0]) + '/' + str(gamma) + '/solarrt']
windfc = stochstore[season + '/' + str(loadhours[0]) + '/' + str(gamma) + '/windfc']
solarfc = stochstore[season + '/' + str(loadhours[0]) + '/' + str(gamma) + '/solarfc']
loadts = stochstore[season + '/' + str(loadhours[0]) + '/' + str(gamma) + '/loadts']
stochstore.close()

print "Data loaded"

scenarios = windrt.items
nda = DA_Nodal(gamma*alpha*windfc, gamma*(1-alpha)*solarfc, loadts)
nrts = [RT_Nodal(gamma*alpha*windrt[s], gamma*(1-alpha)*solarrt[s], loadts) for s in scenarios]

nda.model.setParam('OutputFlag', False)
[nrt.model.setParam('OutputFlag', False) for nrt in nrts]

print "Models Built"

DA_Heuristic = []
RT_Heuristic = []
Labels_Heuristic = []

for loadhour in loadhours:
    stochstore = pd.HDFStore('optimalstore_stoch.h5')
    optimal_cap_df = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_cap']
    bounddf = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_lower_bound']
    windrt = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/windrt']
    solarrt = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/solarrt']
    windfc = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/windfc']
    solarfc = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/solarfc']
    loadts = stochstore[season + '/' + str(loadhour) + '/' + str(gamma) + '/loadts']
    stochstore.close()

    nda._add_new_data(gamma*alpha*windfc, gamma*(1-alpha)*solarfc, loadts)
    nda._update_constraints()
    [nrt._add_new_data(gamma*alpha*windrt[s], gamma*(1-alpha)*solarrt[s], loadts) for s, nrt in zip(scenarios, nrts)]
    [nrt._update_constraints() for s, nrt in zip(scenarios, nrts)]

    ###
    # Set DA dispatch
    ###

    curdispatch = directdispatch.iloc[loadhour]

    for g in nda.data.generators:
        for t in nda.data.taus:
            nda.variables.gprod[g, t].ub = curdispatch[g]
            nda.variables.gprod[g, t].lb = curdispatch[g]
    ####
    # Nodal DA
    ####

    print "Optimize Nodal"

    nda.optimize()
    for nrt in nrts:
        nrt._update_DA_schedule(nda)
        nrt.optimize()

    DA_Heuristic.append(nda.model.ObjVal)
    RT_Heuristic.append(sum(nrt.model.ObjVal for nrt in nrts)/len(nrts))
    Labels_Heuristic.append('Nodal Stochastic')

    print "Nodal Done"

raise SystemExit

resdf = pd.DataFrame({
    'Season': season,
    'Renewable Penetration': gamma,
    'ProcessType': Labels_Heuristic,
    'DA Cost': DA_Heuristic,
    'RT Cost': RT_Heuristic,
    'Hour': loadhours})

outstore = pd.HDFStore('outputstore_stoch.h5')

raise SystemExit

try:
    outstore['results'] = pd.concat((outstore['results'], resdf), ignore_index=True)
except KeyError:
    outstore['results'] = resdf
outstore.close()
