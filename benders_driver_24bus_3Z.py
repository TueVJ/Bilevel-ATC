import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import izip
from collections import defaultdict
from tqdm import tqdm, trange


from benders_bilevel_master_mk2 import Benders_Master


# Loads the 24 bus system, generates the 3 zone graphs. (Figures 2 and 3)

sns.set_style('ticks')

loaddir = '24bus-data_3Z/'

WIND_CAP = 50
HOURS = [1]

load = pd.read_csv(loaddir + 'load.csv')
load = load.set_index('Time')
wind_da = pd.read_csv(loaddir + 'wind.csv')
wind_rt = pd.read_csv(loaddir + 'wind.csv')
wind_rt = wind_rt.set_index(['Time', 'Scenario']).to_panel().transpose(2, 1, 0)*WIND_CAP
wind_da = wind_rt.mean(axis=0)
wind_rt = wind_rt[:, HOURS, :]
wind_da = wind_da.ix[HOURS]
load = load.ix[HOURS]

m = Benders_Master(wind_da, wind_rt, load, loaddir=loaddir)
m.model.setParam('OutputFlag', False)

m.model.Params.IntFeasTol = 1e-08
m.model.Params.VarBranch = 1
m.model.Params.AggFill = 5
m.model.Params.Heuristics = 0
m.model.Params.CutPasses = 3
m.model.Params.Aggregate = 0
m.model.Params.SimplexPricing = 0

print 'Initial Optimization'

m.optimize()

###
# Graph optimal ATCs and expected costs versus line capacity for line 15-24.
###

keys = [(('Z1', 'Z2'), h) for h in HOURS] + [(('Z3', 'Z2'), h) for h in HOURS]

linelimits = np.linspace(0.1, 500, 101)

tm = Benders_Master(wind_da, wind_rt, load, loaddir=loaddir, verbose=False)
tm.model.setParam('OutputFlag', False)
tm.model.params.IntFeasTol = 1e-8

res = []
ATCs = []
for ll in tqdm(linelimits, desc='Graph optimal ATCs for capacity'):
    tm.data.linelimit['n15', 'n24'] = ll
    tm._clear_cuts()
    tm.optimize(force_submodel_rebuild=True)
    res.append(tm.model.ObjVal)
    ATCs.append([tm.variables.ATC[k].x for k in keys])
ATCs = np.array(ATCs).T

plt.figure(figsize=(5, 4))
ax1 = plt.axes()
c = sns.xkcd_rgb['charcoal']
l1 = ax1.plot(linelimits, ATCs[1], '-', c=c, label='ATC, Z1 to Z2')
c = sns.xkcd_rgb['deep red']
l2 = ax1.plot(linelimits, ATCs[0], '-', c=c, label='ATC, Z2 to Z3')
plt.xlabel('Limit on line 15-24 [MW]')
ax1.set_ylabel('ATC [MW]')
ax2 = ax1.twinx()
c = sns.xkcd_rgb['deep blue']
l3 = ax2.plot(linelimits, res, '--', c=c, label='Expected RT cost')
ax2.set_ylabel('Expected total cost [$]')
# plt.grid()
plt.legend([l1[0], l2[0], l3[0]], ['ATC, Z1 to Z2', 'ATC, Z2 to Z3', 'Final expected cost'], loc='center right')
plt.tight_layout()
plt.savefig('../pic/24bus-linecap-vs-ATC.pdf')

###
# Graph expected costs for various ATCs for line capacity of 500 MW for line 15-24.
###

keys = m.variables.ATC.keys()
k1, k2 = keys[0], keys[1]
oatc1, oatc2 = [m.variables.ATC[k].x for k in keys]
oatc1 = min(5000, oatc1)
oatc2 = min(5000, oatc2)

atcs1, atcs2 = np.meshgrid(np.linspace(0.0, 1.5, 21), np.linspace(0.0, 1.5, 21))
atcs1, atcs2 = atcs1*oatc1, atcs2*oatc2
tm = Benders_Master(wind_da, wind_rt, load, loaddir=loaddir, verbose=False)
tm.model.setParam('OutputFlag', False)
tm.model.params.IntFeasTol = 1e-8
res = []
niters = []
for atc1, atc2, _ in izip(atcs1.flat, atcs2.flat, trange(len(atcs1.flatten()), desc='Expected cost vs ATC, 500MW')):
    tm._clear_cuts()
    tm.variables.ATC[k1].ub = atc1
    tm.variables.ATC[k1].lb = atc1
    tm.variables.ATC[k2].ub = atc2
    tm.variables.ATC[k2].lb = atc2
    tm.optimize(force_submodel_rebuild=True)
    res.append(tm.data.lb)
    niters.append(len(tm.data.upper_bounds))

res = np.reshape(res, atcs1.shape)
niters = np.reshape(niters, atcs1.shape)


costlevels = [11600, 12000., 13000., 14000., 15000., 16000., 17000.]
minlevel = 11500
maxlevel = 18500

plt.figure(figsize=(5, 4), dpi=200)
ax = plt.axes()
CF = plt.contourf(atcs1, atcs2, res, 151, cmap=plt.cm.RdYlGn_r, vmin=minlevel, vmax=maxlevel, extend='both')
for cface in CF.collections:
    cface.set_edgecolor("face")
CS = plt.contour(atcs1, atcs2, res, levels=costlevels, colors='k')
plt.clabel(CS, fmt='%1.0f')
plt.scatter([oatc1], [oatc2], c='w')
plt.xlim([atcs1.min(), atcs1.max()])
plt.ylim([atcs2.min(), atcs2.max()])
plt.xlabel('ATC, Zone 1 to Zone 2 [MW]')
plt.ylabel('ATC, Zone 2 to Zone 3 [MW]')
plt.tight_layout()
ax.text(-0.22, 0.98, '(b)', transform=ax.transAxes, size=16)
plt.savefig('../pic/24bus-linecap-500.pdf')


###
# Graph expected costs for various ATCs for line capacity of 150 MW for line 15-24.
###

tm = Benders_Master(wind_da, wind_rt, load, loaddir=loaddir, verbose=False)
tm.model.setParam('OutputFlag', False)
tm.model.params.IntFeasTol = 1e-8
tm.data.linelimit['n15', 'n24'] = 150
tm.optimize()

oatc1, oatc2 = [tm.variables.ATC[k].x for k in keys]
oatc1 = min(5000, oatc1)
oatc2 = min(5000, oatc2)


res2 = []
niters2 = []
for atc1, atc2, _ in izip(atcs1.flat, atcs2.flat, trange(len(atcs1.flatten()), desc='Expected cost vs ATC, 150MW')):
    tm._clear_cuts()
    tm.variables.ATC[k1].ub = atc1
    tm.variables.ATC[k1].lb = atc1
    tm.variables.ATC[k2].ub = atc2
    tm.variables.ATC[k2].lb = atc2
    tm.optimize(force_submodel_rebuild=True)
    res2.append(tm.data.lb)
    niters2.append(len(tm.data.upper_bounds))

res2 = np.reshape(res2, atcs1.shape)
niters2 = np.reshape(niters2, atcs1.shape)

costlevels = [14600, 15000., 16000., 17000., 18000.]
minlevel = 14500
maxlevel = 18500

plt.figure(figsize=(5, 4), dpi=200)
ax2 = plt.axes()
CF = plt.contourf(atcs1, atcs2, res2, 151, cmap=plt.cm.RdYlGn_r, vmin=minlevel, vmax=maxlevel, extend='both')
for cface in CF.collections:
    cface.set_edgecolor("face")
CS = plt.contour(atcs1, atcs2, res2, levels=costlevels, colors='k')
plt.clabel(CS, fmt='%1.0f')
plt.scatter([oatc1], [oatc2], c='w')
plt.xlim([atcs1.min(), atcs1.max()])
plt.ylim([atcs2.min(), atcs2.max()])
plt.xlabel('ATC, Zone 1 to Zone 2 [MW]')
plt.ylabel('ATC, Zone 2 to Zone 3 [MW]')
plt.tight_layout()
ax2.text(-0.22, 0.98, '(a)', transform=ax2.transAxes, size=16)
plt.savefig('../pic/24bus-linecap-150.pdf')
