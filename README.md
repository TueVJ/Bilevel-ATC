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
Generates plots for the 24 bus test system.


## Files not provided

Certain files are not provided due to space limitations. These are:

*EU_data_scenarios/scenariostore*: HDF data store containing signal for load, and scenarios and signals for wind and solar production. The file is organized as

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
