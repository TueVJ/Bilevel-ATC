# Import Gurobi Library
import numpy as np
import gurobipy as gb
import networkx as nx
import defaults
import pickle

from itertools import izip
from collections import defaultdict

from myhelpers import invert_dict, symmetrize_dict, unsymmetrize_list
from load_fnct import load_network, load_generators

from benders_bilevel_subproblem_mk2 import Benders_Subproblem


# Class which can have attributes set.
class expando(object):
    pass


def split_binary(x):
    if x <= 0.5:
        return 0
    return 1


# Optimization class
class Stochastic_Nodal:
    '''
        initial_(wind_da,load) are (N, T) dataframes with columns
        N of nodes in the network, and
        T the timesteps to optimize over.
        initial_wind_rt are (S, N, T) data panels with items S the scenarios for the real time production, major axis T and minor axis N.

        Ex:
        With a single DA point forecast windfc:
        windts = pd.Panel({'s1': windfc})
        yields the correct structure.

    '''
    def __init__(self, initial_wind_da, initial_wind_rt, initial_load, epsilon=0.001, delta=0.001, loaddir='24bus-data_3Z/', verbose=True):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.data.loaddir = loaddir
        self.verbose = verbose
        self._load_data(initial_wind_da, initial_wind_rt, initial_load)
        self._build_model()

    def optimize(self, simple_results=False, force_submodel_rebuild=False):
        # Initial solution
        self.model.optimize()
        pass

    ###
    #   Loading functions
    ###

    def _load_data(self, initial_wind_da, initial_wind_rt, initial_load):
        self._load_network()
        self._load_generator_data()
        self._load_intial_data(initial_wind_da, initial_wind_rt, initial_load)
        self.data.bigM = defaults.bigM

    def _load_network(self):
        self.data.G = load_network(nodefile=self.data.loaddir + 'nodes.csv', linefile=self.data.loaddir + 'lines.csv')
        self.data.nodeorder = self.data.G.nodes()
        self.data.lineorder = self.data.G.edges()
        self.data.linelimit = nx.get_edge_attributes(self.data.G, 'limit')
        self.data.lineadmittance = nx.get_edge_attributes(self.data.G, 'Y')
        self.data.node_to_zone = nx.get_node_attributes(self.data.G, 'country')
        self.data.zone_to_nodes = invert_dict(self.data.node_to_zone)
        self.data.zoneorder = self.data.zone_to_nodes.keys()
        edges = set(
            (z1, z2) for n1, n2 in self.data.lineorder
            for z1 in [self.data.node_to_zone[n1]] for z2 in [self.data.node_to_zone[n2]]
            if z1 != z2)
        # Remove reciprocated entries
        self.data.edgeorder = unsymmetrize_list(list(edges))
        self.data.slack_bus = self.data.nodeorder[0]
        pass

    def _load_generator_data(self):
        self.data.gendf = load_generators(generatorfile=self.data.loaddir + 'generators.csv')
        self.data.generators = self.data.gendf.index
        self.data.generatorinfo = self.data.gendf.T.to_dict()
        self.data.gen_to_node = self.data.gendf.origin.to_dict()
        self.data.node_to_generators = invert_dict(self.data.gen_to_node)
        self.data.gen_to_zone = {g: self.data.node_to_zone[n] for g, n in self.data.gen_to_node.iteritems()}
        self.data.zone_to_generators = invert_dict(self.data.gen_to_zone)
        pass

    def _load_intial_data(self, initial_wind_da, initial_wind_rt, initial_load):
        self.data.taus = initial_load.index
        self.data.scenarios = initial_wind_rt.items
        self.data.scenarioprobs = {s: 1.0/len(self.data.scenarios) for s in self.data.scenarios}
        self.data.load_n_rt = {(n, t): v for n, d in initial_load.to_dict().iteritems() for t, v in d.iteritems()}
        self.data.load_z_da = {
            (z, t): sum(
                self.data.load_n_rt[n, t] for n in self.data.zone_to_nodes[z])
            for z in self.data.zoneorder for t in self.data.taus}
        self.data.load_n_da = {
            (n, t): self.data.load_n_rt[n, t]
            for n in self.data.nodeorder for t in self.data.taus}
        self.data.wind_n_rt = initial_wind_rt
        wind_da_dict = initial_wind_da.to_dict()
        self.data.wind_z_da = {(z, t): sum(wind_da_dict[n][t] for n in self.data.zone_to_nodes[z]) for z in self.data.zoneorder for t in self.data.taus}
        self.data.wind_n_da = {(n, t): wind_da_dict[n][t] for n in self.data.nodeorder for t in self.data.taus}

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        # self.model.setParam('OutputFlag', False)
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):

        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo
        nodes = self.data.nodeorder
        lines = self.data.lineorder
        scenarios = self.data.scenarios
        wind_n_da = self.data.wind_n_da
        load_n_da = self.data.load_n_da
        load_n_rt = self.data.load_n_rt

        m = self.model

        ###
        #  Primal variables
        ###

        # Production of generator g at time t
        self.variables.gprod_da = {}
        for t in taus:
            for g in generators:
                self.variables.gprod_da[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'], name='gen{0} prod at {1}'.format(g, t))
        
        # Renewables and load spilled in node n at time t
        self.variables.winduse_da, self.variables.loadshed_da = {}, {}
        for t in taus:
            for n in nodes:
                self.variables.winduse_da[n, t] = m.addVar(lb=0.0, ub=wind_n_da[n, t], name='z{0} winduse at {1}'.format(n, t))
                self.variables.loadshed_da[n, t] = m.addVar(lb=0.0, ub=load_n_da[n, t], name='z{0} loadshed at {1}'.format(n, t))

        # Power flow on line e
        self.variables.lineflow_da = {}
        for l in lines:
            for t in taus:
                limit = self.data.linelimit[l]
                if limit < 1e-9:
                    limit = gb.GRB.INFINITY
                self.variables.lineflow_da[l, t] = m.addVar(lb=-limit, ub=limit, name='l{0} lineflow at {1}'.format(l, t))

        # Node phase angle at node n at time t
        self.variables.nodeangle_da = {}
        for t in taus:
            for n in nodes:
                self.variables.nodeangle_da[n, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name='theta{0} winduse at {1}'.format(n, t))

        # Production of generator g at time t, up and downregulation
        self.variables.gprod_rt = {}
        self.variables.gprod_rt_up = {}
        self.variables.gprod_rt_down = {}
        for t in taus:
            for g in generators:
                for s in scenarios:
                    self.variables.gprod_rt[g, t, s] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                    self.variables.gprod_rt_up[g, t, s] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                    self.variables.gprod_rt_down[g, t, s] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])

        # Renewables and load spilled in node n at time t
        self.variables.winduse_rt, self.variables.loadshed_rt = {}, {}
        for s in scenarios:
            wind_n_rt = self.data.wind_n_rt[s].to_dict()
            for t in taus:
                for n in nodes:
                    self.variables.winduse_rt[n, t, s] = m.addVar(lb=0.0, ub=wind_n_rt[n][t])
                    self.variables.loadshed_rt[n, t, s] = m.addVar(lb=0.0, ub=load_n_rt[n, t])

        # Power flow on line l
        self.variables.lineflow_rt = {}
        for s in scenarios:
            for l in lines:
                for t in taus:
                    ll = np.where(self.data.linelimit[l] > 1e-9, self.data.linelimit[l], gb.GRB.INFINITY)
                    self.variables.lineflow_rt[l, t, s] = m.addVar(lb=-ll, ub=ll)

        # Voltage angles
        self.variables.nodeangle_rt = {}
        for s in scenarios:
            for t in taus:
                for n in nodes:
                    self.variables.nodeangle_rt[n, t, s] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)

        m.update()
        for t in taus:
            self.variables.nodeangle_da[self.data.slack_bus, t].ub = 0
            self.variables.nodeangle_da[self.data.slack_bus, t].lb = 0
            for s in scenarios:
                self.variables.nodeangle_rt[self.data.slack_bus, t, s].ub = 0
                self.variables.nodeangle_rt[self.data.slack_bus, t, s].lb = 0

    def _build_objective(self):
        taus = self.data.taus
        nodes = self.data.nodeorder

        self.model.setObjective(
            sum(self.data.scenarioprobs[s] *
                (gb.quicksum(
                    self.data.generatorinfo[g]['lincost'] * self.variables.gprod_rt[g, t, s] +
                    defaults.up_redispatch_premium * self.variables.gprod_rt_up[g, t, s] +
                    defaults.down_redispatch_premium * self.variables.gprod_rt_down[g, t, s]
                    for g in self.data.generators for t in taus) +
                gb.quicksum(
                    self.variables.winduse_rt[n, t, s]*defaults.renew_price +
                    self.variables.loadshed_rt[n, t, s]*defaults.VOLL
                    for n in nodes for t in taus))
                for s in self.data.scenarios),
            gb.GRB.MINIMIZE)

    def _build_constraints(self):
        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo
        nodes = self.data.nodeorder
        lines = self.data.lineorder
        scenarios = self.data.scenarios
        wind_n_da = self.data.wind_n_da
        load_n_da = self.data.load_n_da
        bigM = self.data.bigM
        m = self.model

        ###
        # Primal constraints
        ###

        # DA power balance
        self.constraints.powerbalance_da = {}
        for n in nodes:
            for t in taus:
                self.constraints.powerbalance_da[n, t] = m.addConstr(
                    gb.quicksum(self.variables.gprod_da[gen, t] for gen in self.data.node_to_generators[n])
                    + self.variables.loadshed_da[n, t] + self.variables.winduse_da[n, t]
                    - gb.quicksum(self.variables.lineflow_da[l, t] for l in lines if l[0] == n)
                    + gb.quicksum(self.variables.lineflow_da[l, t] for l in lines if l[1] == n),
                    gb.GRB.EQUAL,
                    self.data.load_n_rt[n, t])

        # Coupling to nodal phase angles
        self.constraints.flow_to_angle_da = {}
        for l in lines:
            for t in taus:
                self.constraints.flow_to_angle_da = m.addConstr(
                    self.variables.lineflow_da[l, t],
                    gb.GRB.EQUAL,
                    self.data.lineadmittance[l] * (
                        self.variables.nodeangle_da[l[0], t]
                        - self.variables.nodeangle_da[l[1], t]))

        # RT power balance
        self.constraints.powerbalance_rt = {}
        for s in scenarios:
            for n in nodes:
                for t in taus:
                    self.constraints.powerbalance_rt[n, t, s] = m.addConstr(
                        gb.quicksum(self.variables.gprod_rt[gen, t, s] for gen in self.data.node_to_generators[n])
                        + self.variables.loadshed_rt[n, t, s] + self.variables.winduse_rt[n, t, s]
                        - gb.quicksum(self.variables.lineflow_rt[l, t, s] for l in lines if l[0] == n)
                        + gb.quicksum(self.variables.lineflow_rt[l, t, s] for l in lines if l[1] == n),
                        gb.GRB.EQUAL,
                        self.data.load_n_rt[n, t])

        # Coupling RT flows to node angles
        self.constraints.flow_to_angle_rt = {}
        for s in scenarios:
            for l in lines:
                for t in taus:
                    self.constraints.flow_to_angle_rt[l, t, s] = m.addConstr(
                        self.variables.lineflow_rt[l, t, s],
                        gb.GRB.EQUAL,
                        self.data.lineadmittance[l] * (
                            self.variables.nodeangle_rt[l[0], t, s]
                            - self.variables.nodeangle_rt[l[1], t, s]))

        # Coupling day-ahead and RT production
        self.constraints.DA_to_RT = {}
        for s in scenarios:
            for g in generators:
                for t in taus:
                    self.constraints.DA_to_RT[g, t, s] = m.addConstr(
                        self.variables.gprod_rt[g, t, s],
                        gb.GRB.EQUAL,
                        self.variables.gprod_da[g, t] +
                        self.variables.gprod_rt_up[g, t, s] -
                        self.variables.gprod_rt_down[g, t, s])

        pass
