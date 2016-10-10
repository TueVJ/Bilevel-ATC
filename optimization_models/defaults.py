from os.path import join

# Metadata from dataset
metadata_path = "/home/tue/Output_Data/Metadata/"
metadata_path = "/home/tue/Dropbox/PhD/Papers/ATCs/Python/5s-data/"
nodefile = join(metadata_path, "nodes.csv")
linefile = join(metadata_path, "lines.csv")
generatorfile = join(metadata_path, "generators.csv")
hvdcfile = join(metadata_path, "network_hvdc_links.csv")

# Capacity layout files
caplayout_wind = join(metadata_path, 'wind_layouts.csv')
caplayout_solar = join(metadata_path, 'solar_layouts.csv')

# Directories from which to load and save data
fcdir = '/home/tue/Output_Data/Nodal_FC'
tsdir = '/home/tue/Output_Data/Nodal_TS'


# Files specific to this application
local_metadata_path = 'metadata'
nodeorder_file = join(local_metadata_path, "nodeorder.npy")
lineorder_file = join(local_metadata_path, "lineorder.npy")
generatororder_file = join(local_metadata_path, "generatororder.npy")
zoneorder_file = join(local_metadata_path, 'zoneorder.npy')
zoneedgeorder_file = join(local_metadata_path, 'zoneedgeorder.npy')
hvdcorder_file = join(local_metadata_path, 'hvdcorder.npy')

# Parameters
VOLL = 1000
up_redispatch_premium = 7.90
down_redispatch_premium = 8.59
renew_price = 0
