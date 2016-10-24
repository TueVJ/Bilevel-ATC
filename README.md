# Bilevel-ATC

Packages needed (Tested version):

    - Python 2.7
    - Pandas (0.18.1)
    - Numpy (1.11.0)
    - GurobiPy (6.5.0)
    - NetworkX (1.11)
    - Matplotlib (1.5.1)
    - Seaborn (0.5.1)
    - tqdm (4.8.4)

This software was developed on Ubuntu 16.04, and some syntax may be incompatible with other operating systems.

## File overview

*benders_driver_24bus_3Z.py*:
Generates plots for the 24 bus test system (Figures 2 and 3).

*optimal_driver_stoch.py*:
Optimizes ATCs for the European test system.
Only runs for the hour set in the `loadhour` variable.
To generate the plots in the paper, it is necessary to run this file for `loadhour in [1,...,24]`.

*plot_optimized_ATC_layouts.py*:
Plots the optimized ATC layout (Figure 5) for the set of parameters indicated.
Requires *optimal_driver_stoch.py* to have been run for the same set of parameters.

*heuristic_driver_stoch.py*:
Uses the optimized ATCs, and explicitly evaluates them against all 100 scenarios, along with the heuristics (Static ATC, nodal deterministic).
Requires *optimal_driver_stoch.py*  to have been run for the same set of parameters.


*optimal_driver_stoch_nodal_direct.py*:
Generates the optimal stochastic schedule.

*save_nodal_stochastic_results.py*:
Evaluates the optimal stochastic schedule against all scenarios.
Requires *optimal_driver_stoch_nodal_direct.py* to have been run.

*plot_functions.py*:
Plots Figure 4.
Requires *heuristic_driver_stoch.py*  and to have been run for`loadhour in [1,...,24]`, and requires *save_nodal_stochastic_results.py* to have been run.

## Files not provided

Certain files are not provided due to space limitations. These are:

*EU_data_scenarios/scenariostore.h5*: HDF data store containing signal for load, and scenarios and signals for wind and solar production. The file is organized as

```
class 'pandas.io.pytables.HDFStore'>
File path: EU_data_scenarios/scenariostore.h5
/load/obs                   frame        (shape->[24,1494])
/solar/obs                  frame        (shape->[24,1494])
/solar/pfcs                 frame        (shape->[24,1494])
/solar/scenarios            wide         (shape->[100,24,1494])
/wind/obs                   frame        (shape->[24,1494])
/wind/pfcs                  frame        (shape->[24,1494])
/wind/scenarios             wide         (shape->[100,24,1494])
```

With the contents of `/wind/scenarios` being

```
<class 'pandas.core.panel.Panel'>
Dimensions: 100 (items) x 24 (major_axis) x 1494 (minor_axis)
Items axis: s0 to s99
Major_axis axis: 2013-06-24 00:00:00 to 2013-06-24 23:00:00
Minor_axis axis: 1 to 1514
```

This file is available at **UPLOAD ME**
