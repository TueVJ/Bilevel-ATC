import pandas as pd
from benders_bilevel_master_mk2 import Benders_Master

gamma = 0.5
alpha = 0.75
loadhour = 4
season = 'Summer'

# Load full 5s network
loaddir = '5s-data/'

windfc = pd.read_csv(loaddir + season + '/wind_fc.csv', index_col='Time').ix[[loadhour]]
solarfc = pd.read_csv(loaddir + season + '/solar_fc.csv', index_col='Time').ix[[loadhour]]
loadts = pd.read_csv(loaddir + season + '/load.csv', index_col='Time').ix[[loadhour]]

for df in [windfc, solarfc, loadts]:
    df.columns = map(int, df.columns)

windrt = pd.Panel({'s1': windfc})
solarrt = pd.Panel({'s1': solarfc})

renewfc = gamma*(alpha*windfc + (1-alpha)*solarfc)
renewrt = (gamma*alpha*windrt).add(gamma*(1-alpha)*solarrt)

m = Benders_Master(renewfc, renewrt, loadts, loaddir=loaddir)

m.model.params.MIPFocus = 3
m.model.params.MIPGap = 0.01
m.model.params.Heuristics = 0.20
m.model.params.Cuts = 2
m.model.params.IntFeasTol = 1e-4

m.optimize()


m.model.params.IntFeasTol = 1e-5

m._do_benders_step()
m._do_benders_step()

m.model.params.IntFeasTol = 1e-6

m._do_benders_step()
m._do_benders_step()

m.model.params.IntFeasTol = 1e-7

m._do_benders_step()
m._do_benders_step()

m.model.params.IntFeasTol = 1e-8

m._do_benders_step()
m._do_benders_step()

m.model.params.IntFeasTol = 1e-9

m._do_benders_step()
m._do_benders_step()

m.model.params.MIPGap = 0.0001

m._do_benders_step()
m._do_benders_step()

m.model.params.MIPGap = 0.00001

m._do_benders_step()
# Set NumericFocus at the last step to be
# extra careful about integrality violation.
m.model.params.NumericFocus = 1
m._do_benders_step()
m.model.params.NumericFocus = 0

# Save ATCs found

ks = m.variables.ATC.keys()
optimal_cap_df = pd.DataFrame(
    [m.variables.ATC[k].x for k in ks],
    index=pd.MultiIndex.from_tuples(
        [k[0] for k in ks],
        names=['fromNode', 'toNode'])
)

###
# Bounds!
###
bounddf = pd.DataFrame({
    'upper_bounds': m.data.upper_bounds,
    'lower_bounds': m.data.lower_bounds,
    'mip_gap': m.data.mipgap,
    'solvetime': m.data.solvetime
})

store = pd.HDFStore('optimalstore.h5')
store[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_cap'] = optimal_cap_df
store[season + '/' + str(loadhour) + '/' + str(gamma) + '/optimized_lower_bound'] = bounddf
store.close()
