import pandas as pd

df = pd.read_csv('load_rel.csv')
relload = df.set_index('Time').ix[-1]
tots = pd.read_csv('load_time.csv').set_index('Time')
fulload = pd.DataFrame(data=np.outer(tots['tot'], relload/100), index=tots.index, columns=relload.index)
fulload.to_csv('load.csv', float_format='%.3f')

scenarios = ['s'+str(i) for i in xrange(10)]
times = fulload.index
nodes = fulload.columns
windscs = [(t, n, s, np.random.beta(1, 3)*(n in windnodes)) for t in times for n in nodes for s in scenarios]
winddf = pd.DataFrame(data=windscs, columns=['Time', 'Node', 'Scenario', 'Wind'])
winddf2 = winddf.set_index(['Time', 'Scenario', 'Node']).unstack().reset_index()
winddf2.to_csv('wind.csv', index=False)
# Manual cleanup:
# Remove top line, add 'Time' and 'Scenario' to header.
