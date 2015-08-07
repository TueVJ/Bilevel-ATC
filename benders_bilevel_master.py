# Import Gurobi Library
import gurobipy as gb
import networkx as nx
import defaults
import pickle

from itertools import izip
from collections import defaultdict

from symmetrize_dict import symmetrize_dict, unsymmetrize_list
from load_fnct import load_network, load_generators
from invert_dict import invert_dict

from benders_bilevel_subproblem import Benders_Subproblem

####
#   Benders decomposition via Gurobi + Python
#   Example 3.1 from Conejo et al.'s book on optimization techniques
####


# Class which can have attributes set.
class expando(object):
    pass


# Optimization class
class Benders_Master:
    '''
        initial_(wind,load) are N [(N,t)] arrays where
        N is the number of nodes in the network, [and
        t is the number of timesteps to optimize over.
        Note, that t is fixed by this inital assignment]
    '''
    def __init__(self, initial_wind_da, initial_wind_rt, initial_load, epsilon=0.001):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._init_benders_params(epsilon=epsilon)
        self._load_data(initial_wind_da, initial_wind_rt, initial_load)
        self._build_model()

    def optimize(self, simple_results=False):
        # Initial solution
        self.model.optimize()
        # Build subproblem from solution
        self.submodels = {s: Benders_Subproblem(self, scenario=s) for s in self.data.scenarios}
        # In future, update to list of subproblems
        # OR subproblem class is extended to contain multiple optimization problems
        # (better encapsulation in second approach, better modularity in first).
        [sm.update_fixed_vars(self) for sm in self.submodels.itervalues()]
        [sm.optimize() for sm in self.submodels.itervalues()]
        self._add_cut()
        self._update_bounds()
        self._save_vars()
        self.model.update()
        while self.data.ub > self.data.lb + self.data.epsilon and len(self.data.cutlist) < 10:
            self.model.reset()
            self.model.optimize()
            [sm.update_fixed_vars(self) for sm in self.submodels.itervalues()]
            [sm.optimize() for sm in self.submodels.itervalues()]
            self._add_cut()
            self._update_bounds()
            self._save_vars()
        pass

    ###
    #   Loading functions
    ###

    def _init_benders_params(self, epsilon=0.001):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.alphas = []
        self.data.lambdas = {}
        self.data.epsilon = epsilon
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY

    def _load_data(self, initial_wind_da, initial_wind_rt, initial_load):
        self._load_network()
        self._load_generator_data()
        self._load_intial_data(initial_wind_da, initial_wind_rt, initial_load)
        self.data.bigM = 10000

    def _load_network(self):
        self.data.G = load_network(nodefile='24bus-data_3Z/nodes.csv', linefile='24bus-data_3Z/lines.csv')
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
        self.data.gendf = load_generators(generatorfile='24bus-data_3Z/generators.csv')
        self.data.generators = self.data.gendf.index
        self.data.generatorinfo = self.data.gendf.T.to_dict()
        self.data.gen_to_node = self.data.gendf.origin.to_dict()
        self.data.node_to_generators = invert_dict(self.data.gen_to_node)
        self.data.gen_to_zone = self.data.gendf.country.to_dict()
        self.data.zone_to_generators = invert_dict(self.data.gen_to_zone)
        pass

    def _load_intial_data(self, initial_wind_da, initial_wind_rt, initial_load):
        self.data.taus = [1]  # np.arange(len(initial_load.T))
        self.data.scenarios = ['s0', 's1', 's2', 's3']
        self.data.scenarioprobs = {s: 1.0/len(self.data.scenarios) for s in self.data.scenarios}
        self.data.load_n_rt = initial_load.to_dict()
        self.data.load_z_da = {z: sum(self.data.load_n_rt[n] for n in self.data.zone_to_nodes[z]) for z in self.data.zoneorder}
        self.data.wind_n_rt = initial_wind_rt.to_dict()
        self.data.wind_z_da = {z: sum(initial_wind_da[n] for n in self.data.zone_to_nodes[z]) for z in self.data.zoneorder}

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
        zones = self.data.zoneorder
        edges = self.data.edgeorder
        wind_z_da = self.data.wind_z_da
        load_z_da = self.data.load_z_da

        m = self.model

        ###
        #  Primal variables
        ###

        # Production of generator g at time t
        self.variables.gprod_da = {}
        for t in taus:
            for g in generators:
                self.variables.gprod_da[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'], name='{0} prod at {1}'.format(g, t))

        # Renewables and load spilled in zone z at time t
        self.variables.winduse_da, self.variables.loadshed_da = {}, {}
        for t in taus:
            for z in zones:
                self.variables.winduse_da[z, t] = m.addVar(lb=0.0, ub=wind_z_da[z], name='{0} winduse at {1}'.format(z, t))
                self.variables.loadshed_da[z, t] = m.addVar(lb=0.0, ub=load_z_da[z], name='{0} loadshed at {1}'.format(z, t))

        # Power flow on edge e
        self.variables.edgeflow_da = {}
        for e in edges:
            for t in taus:
                self.variables.edgeflow_da[e, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name='{0} edgeflow at {1}'.format(e, t))

        # Power flow limit on edge e
        self.variables.ATC = {}
        for e in edges:
            for t in taus:
                self.variables.ATC[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='{0} ATC at {1}'.format(e, t))

        ###
        #  Dual variables
        ###

        # # Non-negative duals

        # Generator production limits
        self.variables.d_gprod_da_up = {}
        self.variables.d_gprod_da_down = {}
        for t in taus:
            for g in generators:
                self.variables.d_gprod_da_up[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_gprod_da_down[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)

        # Renewables and load spilled in node at time t
        self.variables.d_winduse_da_up, self.variables.d_loadshed_da_up = {}, {}
        self.variables.d_winduse_da_down, self.variables.d_loadshed_da_down = {}, {}
        for t in taus:
            for z in zones:
                self.variables.d_winduse_da_up[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_winduse_da_down[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_loadshed_da_up[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_loadshed_da_down[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)

        # Power flow on edge e
        self.variables.d_edgeflow_da_up = {}
        self.variables.d_edgeflow_da_down = {}
        for e in edges:
            for t in taus:
                self.variables.d_edgeflow_da_up[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
                self.variables.d_edgeflow_da_down[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)

        # # Unbounded duals

        # Price DA
        self.variables.d_price_da = {}
        for t in taus:
            for z in zones:
                self.variables.d_price_da[z, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)

        ###
        #  Complementarity binary variables
        ###

        # Generation limits
        self.variables.bc_gprod_da_up = {}
        self.variables.bc_gprod_da_down = {}
        for t in taus:
            for g in generators:
                self.variables.bc_gprod_da_up[g, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_gprod_da_down[g, t] = m.addVar(vtype=gb.GRB.BINARY)

        # Renewables and load spilled in node at time t
        self.variables.bc_winduse_da_up, self.variables.bc_loadshed_da_up = {}, {}
        self.variables.bc_winduse_da_down, self.variables.bc_loadshed_da_down = {}, {}
        for t in taus:
            for z in zones:
                self.variables.bc_winduse_da_up[z, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_winduse_da_down[z, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_loadshed_da_up[z, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_loadshed_da_down[z, t] = m.addVar(vtype=gb.GRB.BINARY)

        # Power flow on edge e
        self.variables.bc_edgeflow_da_up = {}
        self.variables.bc_edgeflow_da_down = {}
        for e in edges:
            for t in taus:
                self.variables.bc_edgeflow_da_up[e, t] = m.addVar(vtype=gb.GRB.BINARY)
                self.variables.bc_edgeflow_da_down[e, t] = m.addVar(vtype=gb.GRB.BINARY)

        # Benders' proxy variable
        self.variables.alpha = m.addVar(lb=-10000.0, ub=gb.GRB.INFINITY, name='alpha')

        m.update()

    def _build_objective(self):
        # MISSING: Update this with alpha
        taus = self.data.taus
        zones = self.data.zoneorder

        self.model.setObjective(
            gb.quicksum(
                self.data.generatorinfo[g]['lincost']*self.variables.gprod_da[g, t]
                for g in self.data.generators for t in taus) +
            gb.quicksum(
                self.variables.winduse_da[z, t]*defaults.renew_price +
                self.variables.loadshed_da[z, t]*defaults.VOLL
                for z in zones for t in taus)
            + self.variables.alpha
            + gb.quicksum(0.1*self.variables.ATC[e, t] for e in self.data.edgeorder for t in taus),
            gb.GRB.MINIMIZE)

    def _build_constraints(self):
        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo
        zones = self.data.zoneorder
        edges = self.data.edgeorder
        wind_z_da = self.data.wind_z_da
        load_z_da = self.data.load_z_da
        bigM = self.data.bigM
        m = self.model

        ###
        # Primal constraints
        ###

        # DA power balance
        self.constraints.powerbalance_da = {}
        for z in zones:
            for t in taus:
                self.constraints.powerbalance_da[z, t] = m.addConstr(
                    gb.quicksum(self.variables.gprod_da[gen, t] for gen in self.data.zone_to_generators[z])
                    + self.variables.loadshed_da[z, t] + self.variables.winduse_da[z, t]
                    - gb.quicksum(self.variables.edgeflow_da[e, t] for e in edges if e[0] == z)
                    + gb.quicksum(self.variables.edgeflow_da[e, t] for e in edges if e[1] == z),
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

        ###
        #   DA stationarity
        ###

        # d L / d p_g
        self.constraints.s_da_gen = {}
        for t in taus:
            for g in generators:
                self.constraints.s_da_gen[g, t] = m.addConstr(
                    self.data.generatorinfo[g]['lincost']
                    - self.variables.d_gprod_da_down[g, t]
                    + self.variables.d_gprod_da_up[g, t]
                    - self.variables.d_price_da[self.data.gen_to_zone[g], t],
                    gb.GRB.EQUAL, 0)

        # d L / d w_z
        # d L / d l_z^s
        self.constraints.s_da_wind = {}
        self.constraints.s_da_loadshed = {}
        for t in taus:
            for z in zones:
                self.constraints.s_da_wind[z, t] = m.addConstr(
                    defaults.renew_price
                    - self.variables.d_winduse_da_down[z, t]
                    + self.variables.d_winduse_da_up[z, t]
                    - self.variables.d_price_da[z, t],
                    gb.GRB.EQUAL, 0)
                self.constraints.s_da_loadshed[z, t] = m.addConstr(
                    defaults.VOLL
                    - self.variables.d_loadshed_da_down[z, t]
                    + self.variables.d_loadshed_da_up[z, t]
                    - self.variables.d_price_da[z, t],
                    gb.GRB.EQUAL, 0)

        # d L  / d f_e
        self.constraints.s_da_edgeflow = {}
        for t in taus:
            for e in edges:
                self.constraints.s_da_edgeflow[e, t] = m.addConstr(
                    - self.variables.d_edgeflow_da_down[e, t]
                    + self.variables.d_edgeflow_da_up[e, t]
                    + self.variables.d_price_da[e[0], t]
                    - self.variables.d_price_da[e[1], t],
                    gb.GRB.EQUAL, 0)

        ###
        # Complementary slackness
        ###

        # Generator production limits
        self.constraints.cs_gprod_up_da_p = {}
        self.constraints.cs_gprod_up_da_d = {}
        self.constraints.cs_gprod_down_da_p = {}
        self.constraints.cs_gprod_down_da_d = {}
        for t in taus:
            for g in generators:
                self.constraints.cs_gprod_up_da_d[g, t] = m.addConstr(
                    self.variables.d_gprod_da_up[g, t],
                    gb.GRB.LESS_EQUAL, bigM * self.variables.bc_gprod_da_up[g, t])
                self.constraints.cs_gprod_up_da_p[g, t] = m.addConstr(
                    self.variables.gprod_da[g, t].ub - self.variables.gprod_da[g, t],
                    gb.GRB.LESS_EQUAL, bigM * (1 - self.variables.bc_gprod_da_up[g, t]))
                self.constraints.cs_gprod_down_da_d[g, t] = m.addConstr(
                    self.variables.d_gprod_da_down[g, t],
                    gb.GRB.LESS_EQUAL, bigM * self.variables.bc_gprod_da_down[g, t])
                self.constraints.cs_gprod_down_da_p[g, t] = m.addConstr(
                    self.variables.gprod_da[g, t],
                    gb.GRB.LESS_EQUAL, bigM * (1 - self.variables.bc_gprod_da_down[g, t]))

        # Wind usage limits
        self.constraints.cs_winduse_up_da_p = {}
        self.constraints.cs_winduse_up_da_d = {}
        self.constraints.cs_winduse_down_da_p = {}
        self.constraints.cs_winduse_down_da_d = {}
        for t in taus:
            for z in zones:
                self.constraints.cs_winduse_up_da_d[z, t] = m.addConstr(
                    self.variables.d_winduse_da_up[z, t],
                    gb.GRB.LESS_EQUAL, bigM * self.variables.bc_winduse_da_up[z, t])
                self.constraints.cs_winduse_up_da_p[z, t] = m.addConstr(
                    self.variables.winduse_da[z, t].ub - self.variables.winduse_da[z, t],
                    gb.GRB.LESS_EQUAL, bigM * (1 - self.variables.bc_winduse_da_up[z, t]))
                self.constraints.cs_winduse_down_da_d[z, t] = m.addConstr(
                    self.variables.d_winduse_da_down[z, t],
                    gb.GRB.LESS_EQUAL, bigM * self.variables.bc_winduse_da_down[z, t])
                self.constraints.cs_winduse_down_da_p[z, t] = m.addConstr(
                    self.variables.winduse_da[z, t],
                    gb.GRB.LESS_EQUAL, bigM * (1 - self.variables.bc_winduse_da_down[z, t]))

        # Load shed limits
        self.constraints.cs_loadshed_up_da_p = {}
        self.constraints.cs_loadshed_up_da_d = {}
        self.constraints.cs_loadshed_down_da_p = {}
        self.constraints.cs_loadshed_down_da_d = {}
        for t in taus:
            for z in zones:
                self.constraints.cs_loadshed_up_da_d[z, t] = m.addConstr(
                    self.variables.d_loadshed_da_up[z, t],
                    gb.GRB.LESS_EQUAL, bigM * self.variables.bc_loadshed_da_up[z, t])
                self.constraints.cs_loadshed_up_da_p[z, t] = m.addConstr(
                    self.variables.loadshed_da[z, t].ub - self.variables.loadshed_da[z, t],
                    gb.GRB.LESS_EQUAL, bigM * (1 - self.variables.bc_loadshed_da_up[z, t]))
                self.constraints.cs_loadshed_down_da_d[z, t] = m.addConstr(
                    self.variables.d_loadshed_da_down[z, t],
                    gb.GRB.LESS_EQUAL, bigM * self.variables.bc_loadshed_da_down[z, t])
                self.constraints.cs_loadshed_down_da_p[z, t] = m.addConstr(
                    self.variables.loadshed_da[z, t],
                    gb.GRB.LESS_EQUAL, bigM * (1 - self.variables.bc_loadshed_da_down[z, t]))

        # Edge flow limits
        self.constraints.cs_edgeflow_up_da_p = {}
        self.constraints.cs_edgeflow_up_da_d = {}
        self.constraints.cs_edgeflow_down_da_p = {}
        self.constraints.cs_edgeflow_down_da_d = {}
        for t in taus:
            for e in edges:
                self.constraints.cs_edgeflow_up_da_d[e, t] = m.addConstr(
                    self.variables.d_edgeflow_da_up[e, t],
                    gb.GRB.LESS_EQUAL, bigM * self.variables.bc_edgeflow_da_up[e, t])
                self.constraints.cs_edgeflow_up_da_p[e, t] = m.addConstr(
                    self.variables.ATC[e, t] - self.variables.edgeflow_da[e, t],
                    gb.GRB.LESS_EQUAL, bigM * (1 - self.variables.bc_edgeflow_da_up[e, t]))
                self.constraints.cs_edgeflow_down_da_d[e, t] = m.addConstr(
                    self.variables.d_edgeflow_da_down[e, t],
                    gb.GRB.LESS_EQUAL, bigM * self.variables.bc_edgeflow_da_down[e, t])
                self.constraints.cs_edgeflow_down_da_p[e, t] = m.addConstr(
                    self.variables.ATC[e, t] + self.variables.edgeflow_da[e, t],
                    gb.GRB.LESS_EQUAL, bigM * (1 - self.variables.bc_edgeflow_da_down[e, t]))

        # Bender's optimality cuts
        self.constraints.cuts = {}

        ###
        # SPEED ONLY
        # Up/down constraints cannot be active at same time.
        # ###
        # self.constraints.ud_gprod_da = {}
        # for t in taus:
        #     for g in generators:
        #         self.constraints.ud_gprod_da[g, t] = m.addConstr(
        #             self.variables.bc_gprod_da_up[g, t]
        #             + self.variables.bc_gprod_da_down[g, t],
        #             gb.GRB.LESS_EQUAL, 1.01)

        # self.constraints.ud_winduse_da = {}
        # self.constraints.ud_loadshed_da = {}
        # for t in taus:
        #     for z in zones:
        #         self.constraints.ud_winduse_da[z, t] = m.addConstr(
        #             self.variables.bc_winduse_da_up[z, t]
        #             + self.variables.bc_winduse_da_down[z, t],
        #             gb.GRB.LESS_EQUAL, 1.01)
        #         self.constraints.ud_loadshed_da[z, t] = m.addConstr(
        #             self.variables.bc_loadshed_da_up[z, t]
        #             + self.variables.bc_loadshed_da_down[z, t],
        #             gb.GRB.LESS_EQUAL, 1.01)

        # self.constraints.ud_edgeflow_da = {}
        # for t in taus:
        #     for e in edges:
        #         self.constraints.ud_edgeflow_da[e, t] = m.addConstr(
        #             self.variables.bc_edgeflow_da_up[e, t]
        #             + self.variables.bc_edgeflow_da_down[e, t],
        #             gb.GRB.LESS_EQUAL, 1.01)

        pass

    ###
    # Cut adding
    ###
    def _add_cut(self):

        taus = self.data.taus
        generators = self.data.generators
        zones = self.data.zoneorder

        cut = len(self.data.cutlist)
        self.data.cutlist.append(cut)
        x = self.variables.gprod_da
        # Get sensitivity from subproblem
        sens = {}
        for g in generators:
            for t in taus:
                sens[g, t] = sum(self.data.scenarioprobs[s] * self.submodels[s].constraints.fix_da_production[g, t].pi for s in self.data.scenarios)
        z_sub = sum(self.data.scenarioprobs[s] * self.submodels[s].model.ObjVal for s in self.data.scenarios)
        self.data.lambdas[cut] = sens
        # Generate cut
        self.constraints.cuts[cut] = self.model.addConstr(
            self.variables.alpha,
            gb.GRB.GREATER_EQUAL,
            z_sub
            - sum(
                defaults.VOLL * self.variables.loadshed_da[z, t].x
                + defaults.renew_price * self.variables.winduse_da[z, t].x
                for z in zones for t in taus)
            + gb.quicksum(sens[g, t] * x[g, t] for g in generators for t in taus)
            - sum(sens[g, t] * x[g, t].x for g in generators for t in taus))
        # update model

    def _clear_cuts(self):
        self.data.cutlist = []
        self.data.lambdas = []
        for con in self.constraints.cuts.values():
            self.model.remove(con)
        self.constraints.cuts = {}

    ###
    # Update upper and lower bounds
    ###
    def _update_bounds(self):
        taus = self.data.taus
        generators = self.data.generators
        zones = self.data.zoneorder
        z_sub = sum(self.data.scenarioprobs[s] * self.submodels[s].model.ObjVal for s in self.data.scenarios)
        z_master = self.model.ObjVal
        self.data.ub = \
            z_master - self.variables.alpha.x + z_sub \
            - sum(
                defaults.VOLL * self.variables.loadshed_da[z, t].x
                + defaults.renew_price * self.variables.winduse_da[z, t].x
                for z in zones for t in taus)
        self.data.lb = z_master
        self.data.upper_bounds.append(self.data.ub)
        self.data.lower_bounds.append(self.data.lb)

    def _save_vars(self):
        # self.data.xs.append(self.variables.x.x)
        # self.data.ys.append(self.submodel.variables.y.x)
        self.data.alphas.append(self.variables.alpha.x)

# Temporary testing code

import pandas as pd

load = pd.read_csv('24bus-data_3Z/load.csv')
wind_da = pd.read_csv('24bus-data_3Z/wind.csv')
wind_rt = pd.read_csv('24bus-data_3Z/wind.csv')

load = load.set_index('Time').ix[1]
wind_da = wind_da.set_index('Time').ix[12].set_index('Scenario').mean(axis=0)*100
wind_rt = wind_rt.set_index('Time').ix[12].set_index('Scenario')*100

m = Benders_Master(wind_da, wind_rt, load)

raise SystemExit

import numpy as np

res = {}
atcs = np.linspace(0, 1000, 51)
for atc in atcs:
    m = Benders_Master(wind_da, wind_rt, load)
    for x in m.variables.ATC.values():
        x.ub = atc
        x.lb = atc
    m.optimize()
    res[atc] = m.data.lb

raise SystemExit

# For 3-zone system only
keys = m.variables.ATC.keys()
k1, k2 = keys[0], keys[1]
oatc1, oatc2 = [m.variables.ATC[k].x for k in keys]

atcs1, atcs2 = np.meshgrid(np.linspace(0.0, 2.0, 21), np.linspace(0.0, 2.0, 21))
atcs1, atcs2 = atcs1*oatc1, atcs2*oatc2
res = []
for atc1, atc2 in izip(atcs1.flat, atcs2.flat):
    m = Benders_Master(wind_da, wind_rt, load)
    m.variables.ATC[k1].ub = atc1
    m.variables.ATC[k1].lb = atc1
    m.variables.ATC[k2].ub = atc2
    m.variables.ATC[k2].lb = atc2
    m.optimize()
    res.append(m.data.lb)

res = np.reshape(res, atcs1.shape)

plt.contourf(atcs1, atcs2, res, levels=np.linspace(8000, 10000, 51), cmap=plt.cm.BuPu)
CS = plt.contour(atcs1, atcs2, res, levels np.linspace(8000, 10000, 9), cmap=plt.cm.GnBu_r)
plt.clabel(CS)
