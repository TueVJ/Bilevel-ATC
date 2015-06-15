import numpy as np
import gurobipy as gb
import networkx as nx
import defaults
import pickle
from symmetrize_dict import symmetrize_dict
from itertools import izip
from collections import defaultdict
from load_network import load_network
from load_generators import load_generators
from invert_dict import invert_dict

####
#  Class for the bilevel optimization problem, big-M version
#  Init: Load network, load initial data, build model.
#  Optimize: Optimize the model.
#  15/6: Current implementation does not handle multiple time periods and scenarios
####


# Class which can have attributes set.
class expando(object):
    pass


# Optimization class
class Bilevel_ATC:
    '''
        initial_(wind,load) are N [(N,t)] arrays where
        N is the number of nodes in the network, [and
        t is the number of timesteps to optimize over.
        Note, that t is fixed by this inital assignment]
    '''
    def __init__(self, initial_wind_da, initial_wind_rt, initial_load):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.data = expando()
        self._load_data(initial_wind_da, initial_wind_rt, initial_load)
        self._build_model()

    def optimize(self, simple_results=False):
        self.model.optimize()
        self._build_results()

    ###
    #   Loading functions
    ###

    def _load_data(self, initial_wind_da, initial_wind_rt, initial_load):
        self._load_network()
        self._load_generator_data()
        self._load_intial_data(initial_wind_da, initial_wind_rt, initial_load)
        self.data.bigM = 10000

    def _load_network(self):
        self.data.G = load_network()
        self.data.nodeorder = self.data.G.nodes()
        self.data.lineorder = self.data.G.edges()
        self.data.linelimit = nx.get_edge_attributes(self.data.G, 'limit')
        self.data.node_to_zone = nx.get_node_attributes(self.data.G, 'country')
        self.data.zone_to_nodes = invert_dict(self.data.node_to_zone)
        self.data.zoneorder = self.data.zone_to_nodes.keys()
        edges = set(
            (z1, z2) for n1, n2 in self.data.lineorder
            for z1 in [self.data.node_to_zone[n1]] for z2 in [self.data.node_to_zone[n2]]
            if z1 != z2)
        self.data.edgeorder = list(edges)
        pass

    def _load_generator_data(self):
        self.data.gendf = load_generators()
        self.data.generators = self.data.gendf.index
        self.data.generatorinfo = self.data.gendf.T.to_dict()
        self.data.gen_to_node = self.data.gendf.origin.to_dict()
        self.data.node_to_generators = invert_dict(self.data.gen_to_node)
        self.data.gen_to_zone = self.data.gendf.country.to_dict()
        self.data.zone_to_generators = invert_dict(self.data.gen_to_zone)
        pass

    def _load_intial_data(self, initial_wind_da, initial_wind_rt, initial_load):
        self.data.taus = [0]  # np.arange(len(initial_load.T))
        self.data.load_n_rt = initial_load.to_dict()
        self.data.load_z_da = {z: sum(self.data.load_n_rt[n] for n in self.data.zone_to_nodes[z]) for z in self.data.zoneorder}
        self.data.wind_n_rt = initial_wind_rt.to_dict()
        self.data.wind_z_da = {z: sum(self.data.wind_n_rt[n] for n in self.data.zone_to_nodes[z]) for z in self.data.zoneorder}

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        # self._build_objective()
        # self._build_constraints()
        self.model.update()

    def _build_variables(self):
        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo
        nodes = self.data.nodeorder
        zones = self.data.zoneorder
        edges = self.data.edgeorder
        lines = self.data.lineorder
        wind_n_rt = self.data.wind_n_rt
        wind_z_da = self.data.wind_z_da
        load_n_rt = self.data.load_n_rt
        load_z_da = self.data.load_z_da

        m = self.model

        ###
        #  Primal variables
        ###

        # Production of generator g at time t
        gprod_da = {}
        gprod_rt = {}
        gprod_up = {}
        gprod_down = {}
        for t in taus:
            for g in generators:
                gprod_da[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                gprod_rt[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                gprod_up[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                gprod_down[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
        self.variables.gprod_da = gprod_da
        self.variables.gprod_rt = gprod_rt
        self.variables.gprod_up = gprod_up
        self.variables.gprod_down = gprod_down

        # Renewables and load spilled in node at time t
        renewuse_rt, loadshed_rt = {}, {}
        for t in taus:
            for n in nodes:
                # renewuse[n, t] = m.addVar(lb=0.0, ub=solar[nodeindex[n], t] + wind[nodeindex[n], t])
                # loadshed[n, t] = m.addVar(lb=0.0, ub=load[nodeindex[n], t])
                renewuse_rt[n, t] = m.addVar(lb=0.0, ub=wind_n_rt[n])
                loadshed_rt[n, t] = m.addVar(lb=0.0, ub=load_n_rt[n])
        self.variables.renewuse_rt = renewuse_rt
        self.variables.loadshed_rt = loadshed_rt

        renewuse_da, loadshed_da = {}, {}
        for t in taus:
            for z in zones:
                renewuse_da[z, t] = m.addVar(lb=0.0, ub=wind_z_da[z])
                loadshed_da[z, t] = m.addVar(lb=0.0, ub=load_z_da[z])
        self.variables.renewuse_da = renewuse_da
        self.variables.loadshed_da = loadshed_da

        # Voltage angle at node n for time t
        voltageangle = {}
        for n in nodes:
            for t in taus:
                voltageangle[n, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.voltageangle = voltageangle

        # Power flow on edge e
        edgeflow_da = {}
        for e in edges:
            for t in taus:
                edgeflow_da[e, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.edgeflow_da = edgeflow_da

        # Power flow on line l
        lineflow_rt = {}
        for l in lines:
            for t in taus:
                lineflow_rt[l, t] = m.addVar(lb=-self.data.linelimit[l], ub=self.data.linelimit[l])
        self.variables.lineflow_rt = lineflow_rt

        ###
        #  Dual variables
        ###

        # # Non-negative duals

        # Generator production limits
        d_gprod_da_up = {}
        d_gprod_da_down = {}
        d_gprod_rt_up = {}
        d_gprod_rt_down = {}
        d_gprod_up_nn = {}
        d_gprod_down_nn = {}
        for t in taus:
            for g in generators:
                d_gprod_da_up[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_gprod_rt_up[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_gprod_da_down[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_gprod_rt_down[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_gprod_up_nn[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_gprod_down_nn[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
        self.variables.d_gprod_da_up = d_gprod_da_up
        self.variables.d_gprod_da_down = d_gprod_da_down
        self.variables.d_gprod_rt_up = d_gprod_rt_up
        self.variables.d_gprod_rt_down = d_gprod_rt_down
        self.variables.d_gprod_up_nn = d_gprod_up_nn
        self.variables.d_gprod_down_nn = d_gprod_down_nn

        # Renewables and load spilled in node at time t
        d_renewuse_da_up, d_loadshed_da_up = {}, {}
        d_renewuse_da_down, d_loadshed_da_down = {}, {}
        for t in taus:
            for z in zones:
                d_renewuse_da_up[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_renewuse_da_down[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_loadshed_da_up[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_loadshed_da_down[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
        self.variables.d_renewuse_da_up = d_renewuse_da_up
        self.variables.d_loadshed_da_up = d_loadshed_da_up
        self.variables.d_renewuse_da_down = d_renewuse_da_down
        self.variables.d_loadshed_da_down = d_loadshed_da_down

        d_renewuse_rt_up, d_loadshed_rt_up = {}, {}
        d_renewuse_rt_down, d_loadshed_rt_down = {}, {}
        for t in taus:
            for n in nodes:
                d_renewuse_rt_up[n, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_renewuse_rt_down[n, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_loadshed_rt_up[n, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_loadshed_rt_down[n, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
        self.variables.d_renewuse_rt_up = d_renewuse_rt_up
        self.variables.d_loadshed_rt_up = d_loadshed_rt_up
        self.variables.d_renewuse_rt_down = d_renewuse_rt_down
        self.variables.d_loadshed_rt_down = d_loadshed_rt_down

        # Power flow on edge e
        d_edgeflow_da_up = {}
        d_edgeflow_da_down = {}
        for e in edges:
            for t in taus:
                d_edgeflow_da_up[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_edgeflow_da_down[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
        self.variables.d_edgeflow_da_up = d_edgeflow_da_up
        self.variables.d_edgeflow_da_down = d_edgeflow_da_down

        # Power flow on line l
        d_lineflow_rt_up = {}
        d_lineflow_rt_down = {}
        for l in lines:
            for t in taus:
                d_lineflow_rt_up[l, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                d_lineflow_rt_down[l, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
        self.variables.d_lineflow_rt_up = d_lineflow_rt_up
        self.variables.d_lineflow_rt_down = d_lineflow_rt_down

        # # Unbounded duals

        # Price DA
        d_price_da = {}
        for t in taus:
            for z in zones:
                d_price_da[z, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.d_price_da = d_price_da

        # Price RT
        d_price_rt = {}
        for t in taus:
            for n in nodes:
                d_price_rt[n, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.d_price_rt = d_price_rt

        # Linking day-ahead and real-time production
        d_gprod_link = {}
        for t in taus:
            for g in generators:
                d_gprod_link[g, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.d_gprod_link = d_gprod_link

        # linking to voltage angles
        d_voltage_angle_link = {}
        for l in lines:
            for t in taus:
                d_voltage_angle_link[l, t] = m.addVar(lb=-self.data.linelimit[l], ub=self.data.linelimit[l])
        self.variables.d_voltage_angle_link = d_voltage_angle_link

        # Slack bus dual
        d_slack_bus = {}
        for t in taus:
            d_slack_bus[t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.d_slack_bus = d_slack_bus

        ###
        #  Complementarity binary variables
        ###

        # Generation limits
        bc_gprod_da_up = {}
        bc_gprod_da_down = {}
        bc_gprod_rt_up = {}
        bc_gprod_rt_down = {}
        bc_gprod_up_nn = {}
        bc_gprod_down_nn = {}
        for t in taus:
            for g in generators:
                bc_gprod_da_up[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_gprod_rt_up[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_gprod_da_down[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_gprod_rt_down[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_gprod_up_nn[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_gprod_down_nn[g, t] = m.addVar(vtype=gb.GRB.BINARY)
        self.variables.bc_gprod_da_up = bc_gprod_da_up
        self.variables.bc_gprod_da_down = bc_gprod_da_down
        self.variables.bc_gprod_rt_up = bc_gprod_rt_up
        self.variables.bc_gprod_rt_down = bc_gprod_rt_down
        self.variables.bc_gprod_up_nn = bc_gprod_up_nn
        self.variables.bc_gprod_down_nn = bc_gprod_down_nn

        # Renewables and load spilled in node at time t
        bc_renewuse_da_up, bc_loadshed_da_up = {}, {}
        bc_renewuse_da_down, bc_loadshed_da_down = {}, {}
        for t in taus:
            for z in zones:
                bc_renewuse_da_up[z, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_renewuse_da_down[z, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_loadshed_da_up[z, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_loadshed_da_down[z, t] = m.addVar(vtype=gb.GRB.BINARY)
        self.variables.bc_renewuse_da_up = bc_renewuse_da_up
        self.variables.bc_loadshed_da_up = bc_loadshed_da_up
        self.variables.bc_renewuse_da_down = bc_renewuse_da_down
        self.variables.bc_loadshed_da_down = bc_loadshed_da_down

        bc_renewuse_rt_up, bc_loadshed_rt_up = {}, {}
        bc_renewuse_rt_down, bc_loadshed_rt_down = {}, {}
        for t in taus:
            for n in nodes:
                bc_renewuse_rt_up[n, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_renewuse_rt_down[n, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_loadshed_rt_up[n, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_loadshed_rt_down[n, t] = m.addVar(vtype=gb.GRB.BINARY)
        self.variables.bc_renewuse_rt_up = bc_renewuse_rt_up
        self.variables.bc_loadshed_rt_up = bc_loadshed_rt_up
        self.variables.bc_renewuse_rt_down = bc_renewuse_rt_down
        self.variables.bc_loadshed_rt_down = bc_loadshed_rt_down

        # Power flow on edge e
        bc_edgeflow_da_up = {}
        bc_edgeflow_da_down = {}
        for e in edges:
            for t in taus:
                bc_edgeflow_da_up[e, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_edgeflow_da_down[e, t] = m.addVar(vtype=gb.GRB.BINARY)
        self.variables.bc_edgeflow_da_up = bc_edgeflow_da_up
        self.variables.bc_edgeflow_da_down = bc_edgeflow_da_down

        # Power flow on line l
        bc_lineflow_rt_up = {}
        bc_lineflow_rt_down = {}
        for l in lines:
            for t in taus:
                bc_lineflow_rt_up[l, t] = m.addVar(vtype=gb.GRB.BINARY)
                bc_lineflow_rt_down[l, t] = m.addVar(vtype=gb.GRB.BINARY)
        self.variables.bc_lineflow_rt_up = bc_lineflow_rt_up
        self.variables.bc_lineflow_rt_down = bc_lineflow_rt_down

        m.update()

    def _build_objective(self):
        # MISSING: Update this
        taus = self.data.taus
        nodes = self.data.nodeorder

        m = self.model
        m.setObjective(
            gb.quicksum(self.data.generatorinfo[gen]['lincost']*self.variables.gprod[gen, t] for gen in self.data.generators for t in taus) +
            gb.quicksum(self.variables.renewuse[n, t]*defaults.VOLR + self.variables.loadshed[n, t]*defaults.VOLL for n in nodes for t in taus),
            gb.GRB.MINIMIZE)

    def _build_constraints(self):
        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo
        nodes = self.data.nodeorder
        nodeindex = self.data.nodeindex
        edges = self.data.edgeorder
        wind_da = self.data.wind_da
        wind_rt = self.data.wind_rt
        load = self.data.load

        m = self.model
        voltageangle, nodeinjection, edgeflow = self.variables.voltageangle, self.variables.nodeinjection, self.variables.edgeflow
        loadshed, renewspill, gprod = self.variables.loadshed, self.variables.renewspill, self.variables.gprod

        nodeangle = {}
        for t in taus:
            for n, Bline in izip(nodes, self.data.angle_to_injection):
                nz = np.nonzero(Bline)
                nodeangle[n, t] = m.addConstr(gb.LinExpr(Bline[nz], [voltageangle[k, t] for k in np.array(nodes)[nz]]),
                                              gb.GRB.EQUAL, nodeinjection[n, t], name="CON: node {0} angle from injection in hour {1}".format(n, t))
        self.constraints.nodeangle = nodeangle

        flowangle = {}
        for t in taus:
            for e, Kline in izip(edges, self.data.angle_to_flow):
                nz = np.nonzero(Kline)
                flowangle[e, t] = m.addConstr(gb.LinExpr(Kline[nz], [voltageangle[k, t] for k in nodes[nz]]), gb.GRB.EQUAL, edgeflow[e, t])
        self.constraints.flowangle = flowangle

        powerbalance = {}
        for n in nodes:
            for t in taus:
                powerbalance[n, t] = m.addConstr(nodeinjection[n, t] - loadshed[n, t] + renewspill[n, t]
                                                 - gb.quicksum(gprod[gen, t] for gen in self.data.generatorsfornode[n]),
                                                 gb.GRB.EQUAL,
                                                 solar[nodeindex[n], t] + wind[nodeindex[n], t] - load[nodeindex[n], t])
        self.constraints.powerbalance = powerbalance

        systembalance = {}
        for t in taus:
            systembalance[t] = m.addConstr(gb.quicksum(nodeinjection[n, t] for n in nodes), gb.GRB.EQUAL, 0.0, name='System balance at time {:.00f}'.format(t))
        self.constraints.systembalance = systembalance

    def _build_results(self):
        pass

    ###
    #   Data updating
    ###
    def _add_new_data(self, wind, solar, load):
        pass

    def _update_constraints(self):
        pass

# Temporary testing code

import pandas as pd

load = pd.read_csv('6bus-data/load.csv')
wind_da = pd.read_csv('6bus-data/wind.csv')
wind_rt = pd.read_csv('6bus-data/wind.csv')

load = load.set_index('Time').ix[0]
wind_da = wind_da.set_index('Time').ix[0]
wind_rt = wind_rt.set_index('Time').ix[0]/0.9

m = Bilevel_ATC(wind_da, wind_rt, load)
