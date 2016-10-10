import pandas as pd
import matplotlib.pyplot as plt
from direct_nodal_model import Stochastic_Nodal

gamma = 0.5
alpha = 0.75
loadhour = 0
season = 'Summer'
DELTA_TIME = 24

# Load full 5s network
loaddir = '5s-data/'

windfc = pd.read_csv(loaddir + season + '/wind_fc.csv', index_col='Time')
solarfc = pd.read_csv(loaddir + season + '/solar_fc.csv', index_col='Time')
loadts = pd.read_csv(loaddir + season + '/load.csv', index_col='Time').ix[[0]]

for df in [windfc, solarfc, loadts]:
    df.columns = map(int, df.columns)

windrt = pd.Panel(
    {'s'+str(i): pd.DataFrame(data=windfc.ix[[i]].reset_index(drop=True)) for i in windfc.index}).iloc[-3:]
solarrt = pd.Panel(
    {'s'+str(i): pd.DataFrame(data=solarfc.ix[[i]].reset_index(drop=True)) for i in solarfc.index}).iloc[-3:]

windfc = windrt.mean(axis=0)
solarfc = solarrt.mean(axis=0)
renewfc = gamma*(alpha*windfc + (1-alpha)*solarfc)
renewrt = (gamma*alpha*windrt).add(gamma*(1-alpha)*solarrt)

renewfc = gamma*(alpha*windfc + (1-alpha)*solarfc)
renewrt = (gamma*alpha*windrt).add(gamma*(1-alpha)*solarrt)


store = pd.HDFStore('EU_data_scenarios/scenariostore.h5')
windsc = store['wind/scenarios']
windpfc = store['wind/pfcs']
windobs = store['wind/obs']
solarsc = store['solar/scenarios']
solarpfc = store['solar/pfcs']
solarobs = store['solar/obs']
loadobs = store['load/obs']
store.close()

solarlayout = pd.read_csv('EU_data_scenarios/solar_layouts_COSMO.csv', index_col=0)
windlayout = pd.read_csv('EU_data_scenarios/wind_layouts_COSMO.csv', index_col=0)

solarpfc = solarpfc.multiply(solarlayout.Uniform)
solarsc = pd.Panel({k: df.multiply(solarlayout.Uniform) for k, df in solarsc.iteritems()})
solarobs = solarobs.multiply(solarlayout.Uniform)

windpfc = windpfc.multiply(windlayout.Uniform)
windsc = pd.Panel({k: df.multiply(windlayout.Uniform) for k, df in windsc.iteritems()})
windobs = windobs.multiply(windlayout.Uniform)

renewpfc = gamma*(alpha*windpfc + (1-alpha)*solarpfc)
renewsc = (gamma*alpha*windsc).add(gamma*(1-alpha)*solarsc)

renewsc_c = renewsc.iloc[:10]

renewpfc_c = renewpfc.iloc[loadhour:loadhour+DELTA_TIME]
renewsc_c = renewsc_c.iloc[:, loadhour:loadhour+DELTA_TIME]
loadobs_c = loadobs.iloc[loadhour:loadhour+DELTA_TIME]


m = Stochastic_Nodal(renewpfc_c, renewsc_c, loadobs_c, loaddir=loaddir)

m.model.params.NumericFocus = 1
m.model.params.BarHomogeneous = 1
m.model.params.Presolve = 2
# m.model.params.PreDual = 0
m.optimize()
m.model.params.NumericFocus = 0

raise SystemExit

###
# Save resulting dispatch
###

import defaults

dispatch_DA = pd.DataFrame([[m.variables.gprod_da[g, t].x for g in m.data.generators] for t in m.data.taus], columns=m.data.generators, index=m.data.taus)
cost_per_hour_RT = pd.Series([
    sum(
        m.data.scenarioprobs[s]*(
            sum(m.data.generatorinfo[g]['lincost'] * m.variables.gprod_rt[g, t, s].x for g in m.data.generators) +
            sum(defaults.renew_price * m.variables.winduse_rt[n, t, s].x + defaults.VOLL * m.variables.loadshed_rt[n, t, s].x for n in m.data.nodeorder))
        for s in m.data.scenarios
    )
    for t in m.data.taus], index=m.data.taus)
cost_per_hour_DA = pd.Series([
    sum(m.data.generatorinfo[g]['lincost'] * m.variables.gprod_da[g, t].x for g in m.data.generators) +
    sum(defaults.renew_price * m.variables.winduse_da[n, t].x + defaults.VOLL * m.variables.loadshed_da[n, t].x for n in m.data.nodeorder)
    for t in m.data.taus], index=m.data.taus)

store = pd.HDFStore('stoch_nodal_dispatch.h5')
store['direct_nodal'] = dispatch_DA
store.close()
