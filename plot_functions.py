# coding=UTF8
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')

outstore = pd.HDFStore('outputstore_stoch.h5')
resdf = outstore['results']
outstore.close()

colorcycle = sns.color_palette()

plt.figure(figsize=(6, 4))
ax2 = plt.axes()
heurdf = resdf[resdf['Renewable Penetration'] == 0.5][resdf.Hour == 17].set_index('ProcessType')[['DA Cost', 'RT Cost']].T[['Nodal Stochastic', 'Nodal DA', 'Optimized\n(1% gap)', 'Min/Max flow']].T
(heurdf/1000).rename(index={'Optimized\n(1% gap)':'Optimized'}).plot(kind='bar', ax=ax2, color=colorcycle)
rects = ax2.patches

for i, rect in enumerate(rects):
    h = rect.get_height()
    label = u'{:.00f}'.format(round(h))
    if rect.get_facecolor()[1] > rect.get_facecolor()[2]:
        h = h*1.01
    ax2.text(rect.get_x() + rect.get_width()/2, h + 10, label, ha='center', va='bottom')

c = sns.xkcd_rgb['charcoal']
# plt.axhline(max_lower_bound/1000, ls='--', c=c, alpha=0.7)

plt.xticks(rotation=0)
plt.ylabel(u'Final cost [k$]')
plt.tight_layout()
plt.legend(loc='upper left', ncol=3)
sns.despine()
plt.draw()
plt.ylim([0.9*heurdf.min().min()/1000, 1.10*heurdf.max().max()/1000])
plt.xlabel('')
plt.savefig('../pic/heuristic_comparison.pdf')


plt.figure(figsize=(6, 4), dpi=150)
ax3 = plt.axes()
rtcost = resdf[resdf['Renewable Penetration'] == 0.5].groupby(['ProcessType', 'Hour'])['RT Cost'].mean()
cdf = (rtcost/1000).reset_index().pivot_table(values='RT Cost', index='Hour', columns='ProcessType')[['Nodal Stochastic', 'Nodal DA', 'Optimized\n(1% gap)', 'Min/Max flow']]
cdf = cdf.rename(columns={'Optimized\n(1% gap)': 'Optimized'})
colors = [sns.xkcd_rgb['charcoal'], sns.xkcd_rgb['deep red'], sns.xkcd_rgb['ocean blue'], '#60712F']
markers = ['o', '|', '^', '.']
for (cname, ser), marker, color in zip(cdf.iteritems(), markers, colors):
    ser.plot(label=cname, marker=marker, color=color)
plt.ylabel(u'Expected hourly cost [k$]', fontsize=14)
plt.xlabel(u'Hour', fontsize=14)
plt.legend(loc='upper left')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

plt.tight_layout()
plt.savefig('../pic/Times_costs.pdf')
rtcost_t = cdf

plt.figure(figsize=(6, 4))
ax3 = plt.axes()
rtcost = resdf[resdf['Hour'] == 17].groupby(['ProcessType', 'Renewable Penetration'])['RT Cost'].mean()
cdf = (rtcost/1000).reset_index().pivot_table(values='RT Cost', index='Renewable Penetration', columns='ProcessType')[['Nodal DA', 'Optimized\n(1% gap)', 'Min/Max flow']]
cdf.rename(columns={'Optimized\n(1% gap)': 'Optimized'}).plot(marker='o')
plt.ylabel(u'Expected hourly cost [k$]')
plt.xlabel(u'Renewable Penetration')
plt.savefig('../pic/Penetrations_costs.pdf')
