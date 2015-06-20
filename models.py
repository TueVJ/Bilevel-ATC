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
        self.data.angle_to_injection, self.data.angle_to_flow = self._get_flow_matrices()
        self.data.slack_bus = self.data.nodeorder[0]
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
        self._build_objective()
        self._build_constraints()
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
        self.variables.gprod_da = {}
        self.variables.gprod_rt = {}
        self.variables.gprod_up = {}
        self.variables.gprod_down = {}
        for t in taus:
            for g in generators:
                self.variables.gprod_da[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                self.variables.gprod_rt[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                self.variables.gprod_up[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                self.variables.gprod_down[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])

        # Renewables and load spilled in node at time t
        self.variables.renewuse_rt, self.variables.loadshed_rt = {}, {}
        for t in taus:
            for n in nodes:
                # self.variables.renewuse[n, t] = m.addVar(lb=0.0, ub=solar[nodeindex[n], t] + wind[nodeindex[n], t])
                # self.variables.loadshed[n, t] = m.addVar(lb=0.0, ub=load[nodeindex[n], t])
                self.variables.renewuse_rt[n, t] = m.addVar(lb=0.0, ub=wind_n_rt[n])
                self.variables.loadshed_rt[n, t] = m.addVar(lb=0.0, ub=load_n_rt[n])

        self.variables.renewuse_da, self.variables.loadshed_da = {}, {}
        for t in taus:
            for z in zones:
                self.variables.renewuse_da[z, t] = m.addVar(lb=0.0, ub=wind_z_da[z])
                self.variables.loadshed_da[z, t] = m.addVar(lb=0.0, ub=load_z_da[z])

        # Voltage angle at node n for time t
        self.variables.voltageangle = {}
        for n in nodes:
            for t in taus:
                self.variables.voltageangle[n, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)

        # Power flow on edge e
        self.variables.edgeflow_da = {}
        for e in edges:
            for t in taus:
                self.variables.edgeflow_da[e, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)

        # Power flow on line l
        self.variables.lineflow_rt = {}
        for l in lines:
            for t in taus:
                self.variables.lineflow_rt[l, t] = m.addVar(lb=-self.data.linelimit[l], ub=self.data.linelimit[l])

        # Power flow on edge e
        self.variables.ATC = {}
        for e in edges:
            for t in taus:
                self.variables.ATC[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)

        ###
        #  Dual variables
        ###

        # # Non-negative duals

        # Generator production limits
        self.variables.d_gprod_da_up = {}
        self.variables.d_gprod_da_down = {}
        self.variables.d_gprod_rt_up = {}
        self.variables.d_gprod_rt_down = {}
        self.variables.d_gprod_up_nn = {}
        self.variables.d_gprod_down_nn = {}
        for t in taus:
            for g in generators:
                self.variables.d_gprod_da_up[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_gprod_rt_up[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_gprod_da_down[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_gprod_rt_down[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_gprod_up_nn[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_gprod_down_nn[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)

        # Renewables and load spilled in node at time t
        self.variables.d_renewuse_da_up, self.variables.d_loadshed_da_up = {}, {}
        self.variables.d_renewuse_da_down, self.variables.d_loadshed_da_down = {}, {}
        for t in taus:
            for z in zones:
                self.variables.d_renewuse_da_up[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_renewuse_da_down[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_loadshed_da_up[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_loadshed_da_down[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)

        self.variables.d_renewuse_rt_up, self.variables.d_loadshed_rt_up = {}, {}
        self.variables.d_renewuse_rt_down, self.variables.d_loadshed_rt_down = {}, {}
        for t in taus:
            for n in nodes:
                self.variables.d_renewuse_rt_up[n, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_renewuse_rt_down[n, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_loadshed_rt_up[n, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_loadshed_rt_down[n, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)

        # Power flow on edge e
        self.variables.d_edgeflow_da_up = {}
        self.variables.d_edgeflow_da_down = {}
        for e in edges:
            for t in taus:
                self.variables.d_edgeflow_da_up[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_edgeflow_da_down[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)

        # Power flow on line l
        self.variables.d_lineflow_rt_up = {}
        self.variables.d_lineflow_rt_down = {}
        for l in lines:
            for t in taus:
                self.variables.d_lineflow_rt_up[l, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_lineflow_rt_down[l, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)

        # # Unbounded duals

        # Price DA
        self.variables.d_price_da = {}
        for t in taus:
            for z in zones:
                self.variables.d_price_da[z, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)

        # Price RT
        self.variables.d_price_rt = {}
        for t in taus:
            for n in nodes:
                self.variables.d_price_rt[n, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)

        # Linking day-ahead and real-time production
        self.variables.d_gprod_link = {}
        for t in taus:
            for g in generators:
                self.variables.d_gprod_link[g, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)

        # linking to voltage angles
        self.variables.d_voltage_angle_link = {}
        for l in lines:
            for t in taus:
                self.variables.d_voltage_angle_link[l, t] = m.addVar(lb=-self.data.linelimit[l], ub=self.data.linelimit[l])

        # Slack bus dual
        self.variables.d_slack_bus = {}
        for t in taus:
            self.variables.d_slack_bus[t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)

        ###
        #  Complementarity binary variables
        ###

        # Generation limits
        self.variables.bc_gprod_da_up = {}
        self.variables.bc_gprod_da_down = {}
        self.variables.bc_gprod_rt_up = {}
        self.variables.bc_gprod_rt_down = {}
        self.variables.bc_gprod_up_nn = {}
        self.variables.bc_gprod_down_nn = {}
        for t in taus:
            for g in generators:
                self.variables.bc_gprod_da_up[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_gprod_rt_up[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_gprod_da_down[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_gprod_rt_down[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_gprod_up_nn[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_gprod_down_nn[g, t] = m.addVar(vtype=gb.GRB.BINARY)

        # Renewables and load spilled in node at time t
        self.variables.bc_renewuse_da_up, self.variables.bc_loadshed_da_up = {}, {}
        self.variables.bc_renewuse_da_down, self.variables.bc_loadshed_da_down = {}, {}
        for t in taus:
            for z in zones:
                self.variables.bc_renewuse_da_up[z, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_renewuse_da_down[z, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_loadshed_da_up[z, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_loadshed_da_down[z, t] = m.addVar(vtype=gb.GRB.BINARY)

        self.variables.bc_renewuse_rt_up, self.variables.bc_loadshed_rt_up = {}, {}
        self.variables.bc_renewuse_rt_down, self.variables.bc_loadshed_rt_down = {}, {}
        for t in taus:
            for n in nodes:
                self.variables.bc_renewuse_rt_up[n, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_renewuse_rt_down[n, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_loadshed_rt_up[n, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_loadshed_rt_down[n, t] = m.addVar(vtype=gb.GRB.BINARY)

        # Power flow on edge e
        self.variables.bc_edgeflow_da_up = {}
        self.variables.bc_edgeflow_da_down = {}
        for e in edges:
            for t in taus:
                self.variables.bc_edgeflow_da_up[e, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_edgeflow_da_down[e, t] = m.addVar(vtype=gb.GRB.BINARY)

        # Power flow on line l
        self.variables.bc_lineflow_rt_up = {}
        self.variables.bc_lineflow_rt_down = {}
        for l in lines:
            for t in taus:
                self.variables.bc_lineflow_rt_up[l, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_lineflow_rt_down[l, t] = m.addVar(vtype=gb.GRB.BINARY)

        m.update()

    def _build_objective(self):
        # MISSING: Update this
        taus = self.data.taus
        nodes = self.data.nodeorder

        m = self.model
        m.setObjective(
            gb.quicksum(
                self.data.generatorinfo[g]['lincost']*self.variables.gprod_rt[g, t] +
                defaults.up_redispatch_premium*self.variables.gprod_up[g, t] +
                defaults.down_redispatch_premium*self.variables.gprod_down[g, t]
                for g in self.data.generators for t in taus) +
            gb.quicksum(
                self.variables.renewuse_rt[n, t]*defaults.renew_price +
                self.variables.loadshed_rt[n, t]*defaults.VOLL
                for n in nodes for t in taus),
            gb.GRB.MINIMIZE)

    def _build_constraints(self):
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

        # Flow to voltage angle coupling
        self.constraints.flowangle = {}
        anodes = np.array(nodes)
        for t in taus:
            for l, Kline in izip(lines, self.data.angle_to_flow):
                nz = np.nonzero(Kline)
                self.constraints.flowangle[l, t] = m.addConstr(
                    gb.LinExpr(Kline[nz], [self.variables.voltageangle[n, t] for n in anodes[nz]]),
                    gb.GRB.EQUAL,
                    self.variables.lineflow_rt[l, t])

        # RT power balance
        self.constraints.powerbalance_rt = {}
        for n in nodes:
            for t in taus:
                self.constraints.powerbalance_rt[n, t] = m.addConstr(
                    gb.quicksum(self.variables.gprod_rt[gen, t] for gen in self.data.node_to_generators[n])
                    + self.variables.loadshed_rt[n, t] + self.variables.renewuse_rt[n, t]
                    - gb.quicksum(self.variables.lineflow_rt[l, t] for l in lines if l[0] == n)
                    + gb.quicksum(self.variables.lineflow_rt[l, t] for l in lines if l[1] == n),
                    gb.GRB.EQUAL,
                    self.data.load_n_rt[n])
                # self.data.load_n_rt[n, t]) # !!!

        # Coupling generator production DA and RT
        self.constraints.gencoupling = {}
        for g in generators:
            for t in taus:
                self.constraints.gencoupling[n, t] = m.addConstr(
                    self.variables.gprod_rt[g, t],
                    gb.GRB.EQUAL,
                    self.variables.gprod_da[g, t] + self.variables.gprod_up[g, t] - self.variables.gprod_down[g, t])
                # self.data.load_n_rt[n, t]) # !!!

        # DA power balance
        self.constraints.powerbalance_da = {}
        for z in zones:
            for t in taus:
                self.constraints.powerbalance_da[z, t] = m.addConstr(
                    gb.quicksum(self.variables.gprod_da[gen, t] for gen in self.data.zone_to_generators[z])
                    + self.variables.loadshed_da[z, t] + self.variables.renewuse_da[z, t]
                    - gb.quicksum(self.variables.edgeflow_da[e, t] for e in lines if e[0] == z)
                    + gb.quicksum(self.variables.edgeflow_da[e, t] for e in lines if e[1] == z),
                    gb.GRB.EQUAL,
                    self.data.load_z_da[z])
                # self.data.load_z_da[z, t]) # !!!

        # ATC edge flow limits
        self.constraints.ATC_limit_up = {}
        self.constraints.ATC_limit_down = {}
        for e in edges:
            for t in taus:
                self.constraints.ATC_limit_up[e, t] = m.addConstr(
                    self.variables.edgeflow_da[e, t],
                    gb.GRB.LESS_EQUAL,
                    self.variables.ATC[e, t])
                self.constraints.ATC_limit_down[e, t] = m.addConstr(
                    - self.variables.ATC[e, t],
                    gb.GRB.LESS_EQUAL,
                    self.variables.edgeflow_da[e, t])

        # Slack bus
        self.constraints.slack_bus_const = {}
        for t in taus:
            self.constraints.slack_bus_const[t] = m.addConstr(
                self.variables.voltageangle[self.data.slack_bus, t],
                gb.GRB.EQUAL,
                0)

    def _build_results(self):
        pass

    ###
    #   Data updating
    ###
    def _add_new_data(self, wind, solar, load):
        pass

    def _update_constraints(self):
        pass

    ###
    #   Utility functions
    ###
    def _get_flow_matrices(self):
        G = self.data.G
        nodeorder = self.data.nodeorder
        lineorder = self.data.lineorder
        lap = np.asarray(nx.laplacian_matrix(G, weight='Y', nodelist=nodeorder))
        Xd = symmetrize_dict(nx.get_edge_attributes(G, 'X'))
        Xvec = np.array([Xd[tuple(e)] for e in lineorder])
        K = nx.incidence_matrix(G, oriented=True, nodelist=nodeorder, edgelist=lineorder)
        angle_to_flow = np.squeeze(np.asarray(np.dot(np.diag(1/Xvec), K.T)))
        angle_to_injection = lap
        return angle_to_injection, angle_to_flow

# Temporary testing code

import pandas as pd

load = pd.read_csv('6bus-data/load.csv')
wind_da = pd.read_csv('6bus-data/wind.csv')
wind_rt = pd.read_csv('6bus-data/wind.csv')

load = load.set_index('Time').ix[0]
wind_da = wind_da.set_index('Time').ix[0]
wind_rt = wind_rt.set_index('Time').ix[0]/0.9

m = Bilevel_ATC(wind_da, wind_rt, load)
