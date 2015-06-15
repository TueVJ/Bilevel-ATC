metadata_path = "/media/tue/Data/Dataset/metadata/"
graph_file = metadata_path + "entsoe_2009_v3.gpickle"

zonal_graph_file = metadata_path + 'zonal_model.gpickle'

nodeorder_file = metadata_path + "nodeorder.npy"
edgeorder_file = metadata_path + "edgeorder.npy"
generatororder_file = metadata_path + "generatororder.npy"
zoneorder_file = metadata_path + 'zoneorder.npy'
zoneedgeorder_file = metadata_path + 'zoneedgeorder.npy'

generator_database_file = metadata_path + "generator_database_affiliation.pickle"

VOLL = 3000
VOLR = 20
redispatch_premium = 4
up_redispatch_premium = 7.90
down_redispatch_premium = 8.59

nodearea_pv_file = 'Node_area_PV.npy'
nodearea_wind_off_file = 'Node_area_wind_offshore.npy'
nodearea_wind_on_file = 'Node_area_wind_onshore.npy'

caplayout_wind_uniform = 'wind_layout_uniform.npy'
caplayout_solar_uniform = 'solar_layout_uniform.npy'
caplayout_wind_proportional = 'wind_layout_proportional.npy'
caplayout_solar_proportional = 'solar_layout_proportional.npy'

fcdir = '/media/tue/Data/Dataset/nodal_fc'
tsdir = '/media/tue/Data/Dataset/nodal_ts'
resultdir = '/media/tue/Data/ArticleData/results/'
aggresultdir = '/media/tue/Data/ArticleData/agg_results/'
