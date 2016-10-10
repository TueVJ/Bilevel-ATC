import pandas as pd
import matplotlib.pyplot as plt
from benders_bilevel_master_mk2 import Benders_Master

gamma = 0.8
alpha = 0.75
loadhour = 17
season = 'Summer'
DELTA_TIME = 2

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


m = Benders_Master(renewpfc_c, renewsc_c, loadobs_c, loaddir=loaddir)

m._add_single_ATC_constraints()

m.model.params.MIPFocus = 3
m.model.params.MIPGap = 0.01
m.model.params.Heuristics = 0.20
m.model.params.Cuts = 2
m.model.params.IntFeasTol = 1e-4

m.optimize()

m.model.params.IntFeasTol = 1e-5

m._solve_with_zero_ATCs()
m._do_benders_step()
m._print_benders_info()
m._do_benders_step()
m._print_benders_info()

m.model.params.IntFeasTol = 1e-6

m._solve_with_zero_ATCs()
m._do_benders_step()
m._print_benders_info()
m._do_benders_step()
m._print_benders_info()

m.model.params.IntFeasTol = 1e-7

m._solve_with_zero_ATCs()
m._do_benders_step()
m._print_benders_info()
m._do_benders_step()
m._print_benders_info()

m.model.params.IntFeasTol = 1e-8

m._solve_with_zero_ATCs()
m._do_benders_step()
m._print_benders_info()
m._do_benders_step()
m._print_benders_info()

m.model.params.IntFeasTol = 1e-9

m._solve_with_zero_ATCs()
m._do_benders_step()
m._print_benders_info()
m._do_benders_step()
m._print_benders_info()

m.model.params.MIPGap = 0.0001

m._do_benders_step()
m._print_benders_info()
m._do_benders_step()
m._print_benders_info()

m.model.params.MIPGap = 0.00001

m._do_benders_step()
m._print_benders_info()
# Set NumericFocus at the last step to be
# extra careful about integrality violation.
m.model.params.NumericFocus = 1
m._do_benders_step()
m._print_benders_info()
m.model.params.NumericFocus = 0

raise SystemExit

# Save ATCs found

optimal_cap_df = pd.DataFrame(
    [[m.variables.ATC[e, t].x for t in m.data.taus] for e in m.data.edgeorder],
    index=pd.MultiIndex.from_tuples(m.data.edgeorder, names=['fromNode', 'toNode']),
    columns=m.data.taus)

###
# Bounds!
###
bounddf = pd.DataFrame({
    'upper_bounds': m.data.upper_bounds,
    'lower_bounds': m.data.lower_bounds,
    'mip_gap': m.data.mipgap,
    'solvetime': m.data.solvetime
})

raise SystemExit

store = pd.HDFStore('optimalstore_stoch.h5')
store[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_cap'] = optimal_cap_df
store[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_lower_bound'] = bounddf
store[season + '/' + str(loadhour) + '/' + str(gamma) + '/windrt'] = windsc.iloc[:, loadhour:loadhour+DELTA_TIME]
store[season + '/' + str(loadhour) + '/' + str(gamma) + '/solarrt'] = solarsc.iloc[:, loadhour:loadhour+DELTA_TIME]
store[season + '/' + str(loadhour) + '/' + str(gamma) + '/windfc'] = windpfc.iloc[loadhour:loadhour+DELTA_TIME]
store[season + '/' + str(loadhour) + '/' + str(gamma) + '/solarfc'] = solarpfc.iloc[loadhour:loadhour+DELTA_TIME]
store[season + '/' + str(loadhour) + '/' + str(gamma) + '/loadts'] = loadobs_c
try:
    olddf = store['/'.join((season, str(gamma), 'ATCdf'))]
    olddf[optimal_cap_df.columns] = optimal_cap_df
    store['/'.join((season, str(gamma), 'ATCdf'))] = olddf
except KeyError:
    store['/'.join((season, str(gamma), 'ATCdf'))] = optimal_cap_df
store.close()
