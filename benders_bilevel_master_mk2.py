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
class Benders_Master:
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
        self._init_benders_params(epsilon=epsilon, delta=delta)
        self._load_data(initial_wind_da, initial_wind_rt, initial_load)
        self._build_model()

    def optimize(self, simple_results=False, force_submodel_rebuild=False):
        # Initial solution
        self.model.optimize()
        # Only build submodels if they don't exist or a rebuild is forced.
        print 'Build submodels'
        if not hasattr(self, 'submodels') or force_submodel_rebuild:
            self.submodels = {s: Benders_Subproblem(self, scenario=s) for s in self.data.scenarios}
        print 'Update fixed vars'
        [sm.update_fixed_vars(self) for sm in self.submodels.itervalues()]
        print 'Optimize Submodels'
        [sm.optimize() for sm in self.submodels.itervalues()]
        print 'Update bounds'
        self._update_bounds()
        self._save_vars()
        self._add_cut()
        # Build cuts until we reach absolute and relative tolerance, or 10 cuts have been generated.
        while (
                self.data.ub > self.data.lb + self.data.delta or
                self.data.ub - self.data.lb > abs(self.data.epsilon * self.data.lb)) and \
                len(self.data.cutlist) < 25:
            if self.verbose:
                self._print_benders_info()
            self._do_benders_step()
        pass

    def _do_benders_step(self):
            # self.model.update()
            self._start_from_previous()
            self.model.optimize()
            [sm.update_fixed_vars(self) for sm in self.submodels.itervalues()]
            [sm.optimize() for sm in self.submodels.itervalues()]
            self._update_bounds()
            self._save_vars()
            self._add_cut()

    def _print_benders_info(self):
        print('')
        print('********')
        print('* Benders\' step {0}:'.format(len(self.data.upper_bounds)))
        print('* Upper bound: {0}'.format(self.data.ub))
        print('* Lower bound: {0}'.format(self.data.lb))
        print('********')
        print('')

    ###
    #   Loading functions
    ###

    def _init_benders_params(self, epsilon=0.001, delta=0.001):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.mipgap = []
        self.data.solvetime = []
        self.data.alphas = []
        self.data.lambdas = {}
        self.data.epsilon = epsilon
        self.data.delta = delta
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        # Used to rescale alpha -> alpha/alphascalefactor to avoid numerical difficulties.
        self.data.alphascalefactor = 100

    def _load_data(self, initial_wind_da, initial_wind_rt, initial_load):
        self._load_network()
        self._load_generator_data()
        self._load_intial_data(initial_wind_da, initial_wind_rt, initial_load)
        self.data.bigM = defaults.bigM
        self.data.ATCweight = 0.000001

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
        self.data.wind_n_rt = initial_wind_rt
        wind_da_dict = initial_wind_da.to_dict()
        self.data.wind_z_da = {(z, t): sum(wind_da_dict[n][t] for n in self.data.zone_to_nodes[z]) for z in self.data.zoneorder for t in self.data.taus}

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
                self.variables.gprod_da[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'], name='gen{0} prod at {1}'.format(g, t))
        # !
        # Renewables and load spilled in zone z at time t
        self.variables.winduse_da, self.variables.loadshed_da = {}, {}
        for t in taus:
            for z in zones:
                self.variables.winduse_da[z, t] = m.addVar(lb=0.0, ub=wind_z_da[z, t], name='z{0} winduse at {1}'.format(z, t))
                self.variables.loadshed_da[z, t] = m.addVar(lb=0.0, ub=load_z_da[z, t], name='z{0} loadshed at {1}'.format(z, t))

        # Power flow on edge e
        self.variables.edgeflow_da = {}
        for e in edges:
            for t in taus:
                self.variables.edgeflow_da[e, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name='zl{0} edgeflow at {1}'.format(e, t))

        # Power flow limit on edge e
        self.variables.ATC = {}
        for e in edges:
            for t in taus:
                self.variables.ATC[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='zl{0} ATC at {1}'.format(e, t))

        ###
        #  Dual variables
        ###

        # # Non-negative duals

        # Generator production limits
        self.variables.d_gprod_da_up = {}
        self.variables.d_gprod_da_down = {}
        for t in taus:
            for g in generators:
                self.variables.d_gprod_da_up[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='d_up_gen{0} prod at {1}'.format(g, t))
                self.variables.d_gprod_da_down[g, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='d_down_gen{0} prod at {1}'.format(g, t))

        # Renewables and load spilled in node at time t
        self.variables.d_winduse_da_up, self.variables.d_loadshed_da_up = {}, {}
        self.variables.d_winduse_da_down, self.variables.d_loadshed_da_down = {}, {}
        for t in taus:
            for z in zones:
                self.variables.d_winduse_da_up[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='d_up_wind_z{0} at {1}'.format(z, t))
                self.variables.d_winduse_da_down[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='d_down_wind_z{0} at {1}'.format(z, t))
                self.variables.d_loadshed_da_up[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='d_up_loadshed_z{0} at {1}'.format(z, t))
                self.variables.d_loadshed_da_down[z, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='d_down_loadshed_z{0} at {1}'.format(z, t))

        # Power flow on edge e
        self.variables.d_edgeflow_da_up = {}
        self.variables.d_edgeflow_da_down = {}
        for e in edges:
            for t in taus:
                self.variables.d_edgeflow_da_up[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='d_up_zl{0} edgeflow at {1}'.format(e, t))
                self.variables.d_edgeflow_da_down[e, t] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='d_down_zl{0} edgeflow at {1}'.format(e, t))

        # # Unbounded duals

        # Price DA
        self.variables.d_price_da = {}
        for t in taus:
            for z in zones:
                self.variables.d_price_da[z, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name='da_price_z{0} at {1}'.format(z, t))

        ###
        #  Complementarity binary variables
        ###

        # Generation limits
        self.variables.bc_gprod_da_up = {}
        self.variables.bc_gprod_da_down = {}
        for t in taus:
            for g in generators:
                self.variables.bc_gprod_da_up[g, t] = m.addVar(vtype=gb.GRB.BINARY, name='b_up_gen{0} prod at {1}'.format(g, t))
                self.variables.bc_gprod_da_down[g, t] = m.addVar(vtype=gb.GRB.BINARY, name='b_down_gen{0} prod at {1}'.format(g, t))

        # Renewables and load spilled in node at time t
        self.variables.bc_winduse_da_up, self.variables.bc_loadshed_da_up = {}, {}
        self.variables.bc_winduse_da_down, self.variables.bc_loadshed_da_down = {}, {}
        for t in taus:
            for z in zones:
                self.variables.bc_winduse_da_up[z, t] = m.addVar(vtype=gb.GRB.BINARY, name='b_up_wind_z{0} at {1}'.format(z, t))
                self.variables.bc_winduse_da_down[z, t] = m.addVar(vtype=gb.GRB.BINARY, name='b_down_wind_z{0} at {1}'.format(z, t))
                self.variables.bc_loadshed_da_up[z, t] = m.addVar(vtype=gb.GRB.BINARY, name='b_up_loadshed_z{0} at {1}'.format(z, t))
                self.variables.bc_loadshed_da_down[z, t] = m.addVar(vtype=gb.GRB.BINARY, name='b_down_loadshed_z{0} at {1}'.format(z, t))

        # Power flow on edge e
        self.variables.bc_edgeflow_da_up = {}
        self.variables.bc_edgeflow_da_down = {}
        for e in edges:
            for t in taus:
                self.variables.bc_edgeflow_da_up[e, t] = m.addVar(vtype=gb.GRB.BINARY, name='b_up_zl{0} edgeflow at {1}'.format(e, t))
                self.variables.bc_edgeflow_da_down[e, t] = m.addVar(vtype=gb.GRB.BINARY, name='b_down_zl{0} edgeflow at {1}'.format(e, t))

        # Benders' proxy variable
        # self.variables.alpha = m.addVar(lb=-10000.0, ub=gb.GRB.INFINITY, name='alpha')

        self.variables.alpha = {}
        for s in self.data.scenarios:
            self.variables.alpha[s] = m.addVar(lb=-1000000.0, ub=gb.GRB.INFINITY, name='alpha_{0}'.format(s))

        # Optimization variable (Used to hint at the lower bound)

        self.variables.objvar = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name='ObjVar')
        m.update()

    def _build_objective(self):
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
            + gb.quicksum(self.data.scenarioprobs[s] * self.variables.alpha[s] for s in self.data.scenarios)
            # + self.variables.alpha
            + gb.quicksum(self.data.ATCweight*self.variables.ATC[e, t] for e in self.data.edgeorder for t in taus),
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

        # !
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
                    self.data.load_z_da[z, t])

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
                    gb.GRB.LESS_EQUAL, 10 * bigM * self.variables.bc_gprod_da_up[g, t])
                self.constraints.cs_gprod_up_da_p[g, t] = m.addConstr(
                    self.variables.gprod_da[g, t].ub - self.variables.gprod_da[g, t],
                    gb.GRB.LESS_EQUAL, bigM * (1 - self.variables.bc_gprod_da_up[g, t]))
                self.constraints.cs_gprod_down_da_d[g, t] = m.addConstr(
                    self.variables.d_gprod_da_down[g, t],
                    gb.GRB.LESS_EQUAL, 10 * bigM * self.variables.bc_gprod_da_down[g, t])
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
                    gb.GRB.LESS_EQUAL, 10 * bigM * self.variables.bc_winduse_da_up[z, t])
                self.constraints.cs_winduse_up_da_p[z, t] = m.addConstr(
                    self.variables.winduse_da[z, t].ub - self.variables.winduse_da[z, t],
                    gb.GRB.LESS_EQUAL, bigM * (1 - self.variables.bc_winduse_da_up[z, t]))
                self.constraints.cs_winduse_down_da_d[z, t] = m.addConstr(
                    self.variables.d_winduse_da_down[z, t],
                    gb.GRB.LESS_EQUAL, 10 * bigM * self.variables.bc_winduse_da_down[z, t])
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
                    gb.GRB.LESS_EQUAL, 10 * bigM * self.variables.bc_loadshed_da_up[z, t])
                self.constraints.cs_loadshed_up_da_p[z, t] = m.addConstr(
                    self.variables.loadshed_da[z, t].ub - self.variables.loadshed_da[z, t],
                    gb.GRB.LESS_EQUAL, self.variables.loadshed_da[z, t].ub * (1 - self.variables.bc_loadshed_da_up[z, t]))
                self.constraints.cs_loadshed_down_da_d[z, t] = m.addConstr(
                    self.variables.d_loadshed_da_down[z, t],
                    gb.GRB.LESS_EQUAL, 10 * bigM * self.variables.bc_loadshed_da_down[z, t])
                self.constraints.cs_loadshed_down_da_p[z, t] = m.addConstr(
                    self.variables.loadshed_da[z, t],
                    gb.GRB.LESS_EQUAL, self.variables.loadshed_da[z, t].ub * (1 - self.variables.bc_loadshed_da_down[z, t]))

        # Edge flow limits
        self.constraints.cs_edgeflow_up_da_p = {}
        self.constraints.cs_edgeflow_up_da_d = {}
        self.constraints.cs_edgeflow_down_da_p = {}
        self.constraints.cs_edgeflow_down_da_d = {}
        for t in taus:
            for e in edges:
                self.constraints.cs_edgeflow_up_da_d[e, t] = m.addConstr(
                    self.variables.d_edgeflow_da_up[e, t],
                    gb.GRB.LESS_EQUAL, 10 * bigM * self.variables.bc_edgeflow_da_up[e, t])
                self.constraints.cs_edgeflow_up_da_p[e, t] = m.addConstr(
                    self.variables.ATC[e, t] - self.variables.edgeflow_da[e, t],
                    gb.GRB.LESS_EQUAL, 10 * bigM * (1 - self.variables.bc_edgeflow_da_up[e, t]))
                self.constraints.cs_edgeflow_down_da_d[e, t] = m.addConstr(
                    self.variables.d_edgeflow_da_down[e, t],
                    gb.GRB.LESS_EQUAL, 10 * bigM * self.variables.bc_edgeflow_da_down[e, t])
                self.constraints.cs_edgeflow_down_da_p[e, t] = m.addConstr(
                    self.variables.ATC[e, t] + self.variables.edgeflow_da[e, t],
                    gb.GRB.LESS_EQUAL, 10 * bigM * (1 - self.variables.bc_edgeflow_da_down[e, t]))

        # Bender's optimality cuts
        self.constraints.cuts = {}

        # ###
        # # SPEED ONLY
        # # Up/down constraints cannot be active at same time.
        # # ###
        self.constraints.ud_gprod_da = {}
        for t in taus:
            for g in generators:
                self.constraints.ud_gprod_da[g, t] = m.addConstr(
                    self.variables.bc_gprod_da_up[g, t]
                    + self.variables.bc_gprod_da_down[g, t],
                    gb.GRB.LESS_EQUAL, 1.0)

        self.constraints.ud_winduse_da = {}
        self.constraints.ud_loadshed_da = {}
        for t in taus:
            for z in zones:
                self.constraints.ud_winduse_da[z, t] = m.addConstr(
                    self.variables.bc_winduse_da_up[z, t]
                    + self.variables.bc_winduse_da_down[z, t],
                    gb.GRB.LESS_EQUAL, 1.0)
                self.constraints.ud_loadshed_da[z, t] = m.addConstr(
                    self.variables.bc_loadshed_da_up[z, t]
                    + self.variables.bc_loadshed_da_down[z, t],
                    gb.GRB.LESS_EQUAL, 1.0)

        self.constraints.ud_edgeflow_da = {}
        for t in taus:
            for e in edges:
                self.constraints.ud_edgeflow_da[e, t] = m.addConstr(
                    self.variables.bc_edgeflow_da_up[e, t]
                    + self.variables.bc_edgeflow_da_down[e, t],
                    gb.GRB.LESS_EQUAL, 1.0)

        ###
        # SPEED ONLY
        # We have strict merit-order activation of units in each zone.
        # Found to slow down dramatically with IntFeasTol<10^-5
        # ###
        # self.constraints.b_meritorder_gprod_da = {}
        # for t in taus:
        #     for z in zones:
        #         gens = self.data.zone_to_generators[z]
        #         # Sort by merit order
        #         gens = sorted(gens, key=lambda x: self.data.generatorinfo[x]['lincost'])
        #         for glow, ghigh in izip(gens[:-1], gens[1:]):
        #             # If glow is not at upper bound, don't activate ghigh
        #             self.constraints.b_meritorder_gprod_da[z, t, glow, ghigh] = m.addConstr(
        #                 1 - self.variables.bc_gprod_da_up[glow, t],
        #                 gb.GRB.LESS_EQUAL, self.variables.bc_gprod_da_down[ghigh, t])
        #             self.constraints.b_meritorder_gprod_da[z, t, glow, ghigh].lazy = True

        pass

    ###
    # Cut adding
    # ###

    # def _add_cut(self):

    #     taus = self.data.taus
    #     generators = self.data.generators
    #     zones = self.data.zoneorder

    #     cut = len(self.data.cutlist)
    #     self.data.cutlist.append(cut)
    #     x = self.variables.gprod_da
    #     # Get sensitivity from subproblem
    #     sens = {}
    #     for g in generators:
    #         for t in taus:
    #             sens[g, t] = sum(self.data.scenarioprobs[s] * self.submodels[s].constraints.fix_da_production[g, t].pi for s in self.data.scenarios)
    #     z_sub = sum(self.data.scenarioprobs[s] * self.submodels[s].model.ObjVal for s in self.data.scenarios)
    #     self.data.lambdas[cut] = sens
    #     # Generate cut
    #     self.constraints.cuts[cut] = self.model.addConstr(
    #         self.variables.alpha,
    #         gb.GRB.GREATER_EQUAL,
    #         z_sub
    #         - sum(
    #             defaults.VOLL * self.variables.loadshed_da[z, t].x
    #             + defaults.renew_price * self.variables.winduse_da[z, t].x
    #             for z in zones for t in taus)
    #         + gb.quicksum(sens[g, t] * x[g, t] for g in generators for t in taus)
    #         - sum(sens[g, t] * x[g, t].x for g in generators for t in taus))
    #     # update model

    # LEGACY: Attempt to use multicuts
    def _add_cut(self):

        taus = self.data.taus
        generators = self.data.generators
        zones = self.data.zoneorder

        cutno = len(self.data.cutlist)
        self.data.cutlist.append(cutno)
        x = self.variables.gprod_da
        # Construct Benders' multicuts
        for s in self.data.scenarios:
            # Get sensitivity from subproblem
            sens = {}
            for g in generators:
                for t in taus:
                    sens[g, t] = self.submodels[s].constraints.fix_da_production[g, t].pi
            z_sub = self.submodels[s].model.ObjVal
            self.data.lambdas[cutno, s] = sens
            # Generate cut
            self.constraints.cuts[cutno, s] = self.model.addConstr(
                self.variables.alpha[s],
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
        self.data.lambdas = {}
        self.model.update()
        for con in self.constraints.cuts.values():
            self.model.remove(con)
        self.constraints.cuts = {}
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        self.data.upper_bounds = []
        self.data.lower_bounds = []

    ###
    # Update upper and lower bounds for Benders' iterations
    ###
    def _update_bounds(self):
        taus = self.data.taus
        generators = self.data.generators
        zones = self.data.zoneorder
        z_sub = sum(self.data.scenarioprobs[s] * self.submodels[s].model.ObjVal for s in self.data.scenarios)
        z_master = self.model.ObjVal - sum(self.data.ATCweight*self.variables.ATC[e, t].x for e in self.data.edgeorder for t in taus)
        # self.data.ub = \
        #     z_master - self.variables.alpha.x + z_sub \
        #     - sum(
        #         defaults.VOLL * self.variables.loadshed_da[z, t].x
        #         + defaults.renew_price * self.variables.winduse_da[z, t].x
        #         for z in zones for t in taus)
        self.data.ub = \
            z_master - sum(self.data.scenarioprobs[s] * self.variables.alpha[s].x for s in self.data.scenarios) + z_sub \
            - sum(
                defaults.VOLL * self.variables.loadshed_da[z, t].x
                + defaults.renew_price * self.variables.winduse_da[z, t].x
                for z in zones for t in taus)
        # self.data.ub = sum(self.data.generatorinfo[g]['lincost']*self.variables.gprod_da[g, t].x for g in self.data.generators for t in taus) + \
        #     sum(self.data.scenarioprobs[s] * self.submodels[s].model.ObjVal for s in self.data.scenarios)
        # The best lower bound is the current bestbound,
        # This will equal z_master at optimality
        self.data.lb = self.model.ObjBound - sum(self.data.ATCweight*self.variables.ATC[e, t].x for e in self.data.edgeorder for t in taus)
        self.data.upper_bounds.append(self.data.ub)
        self.data.lower_bounds.append(self.data.lb)
        self.data.mipgap.append(self.model.params.IntFeasTol)
        self.data.solvetime.append(self.model.Runtime)

    def _save_vars(self):
        self.data.alphas.append(self.variables.alpha)

    ###
    # Check complementarity constraints
    ###
    def _fix_complementarity(self):
        self._calculate_complementarity()
        self._reduce_nonzero_bigMs()

    def _calculate_complementarity(self):
        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo
        zones = self.data.zoneorder
        edges = self.data.edgeorder
        bigM = self.data.bigM

        # Generator production
        self.results.cs_gprod_up_da = {}
        self.results.cs_gprod_down_da = {}
        for t in taus:
            for g in generators:
                self.results.cs_gprod_up_da[g, t] = \
                    self.variables.d_gprod_da_up[g, t].x * (self.variables.gprod_da[g, t].ub - self.variables.gprod_da[g, t].x)
                self.results.cs_gprod_down_da[g, t] = \
                    self.variables.d_gprod_da_down[g, t].x * self.variables.gprod_da[g, t].x

        # Wind usage
        self.results.cs_winduse_up_da = {}
        self.results.cs_winduse_down_da = {}
        for t in taus:
            for z in zones:
                self.results.cs_winduse_up_da[z, t] = \
                    self.variables.d_winduse_da_up[z, t].x * (self.variables.winduse_da[z, t].ub - self.variables.winduse_da[z, t].x)
                self.results.cs_winduse_down_da[z, t] = self.variables.d_winduse_da_down[z, t].x * self.variables.winduse_da[z, t].x

        # Load shed limits
        self.results.cs_loadshed_up_da = {}
        self.results.cs_loadshed_down_da = {}
        for t in taus:
            for z in zones:
                self.results.cs_loadshed_up_da[z, t] = \
                    self.variables.d_loadshed_da_up[z, t].x * (self.variables.loadshed_da[z, t].ub - self.variables.loadshed_da[z, t].x)
                self.results.cs_loadshed_down_da[z, t] = \
                    self.variables.d_loadshed_da_down[z, t].x * self.variables.loadshed_da[z, t].x

        # Edge flow limits
        self.constraints.cs_edgeflow_up_da = {}
        self.constraints.cs_edgeflow_down_da = {}
        for t in taus:
            for e in edges:
                self.constraints.cs_edgeflow_up_da[e, t] = \
                    self.variables.d_edgeflow_da_up[e, t].x * (self.variables.ATC[e, t].x - self.variables.edgeflow_da[e, t])
                self.constraints.cs_edgeflow_down_da[e, t] = \
                    self.variables.d_edgeflow_da_down[e, t].x * (self.variables.ATC[e, t].x + self.variables.edgeflow_da[e, t])

    def _reduce_nonzero_bigMs(self):
        pass

    ###
    # MIP start conditions
    ###
    def _start_from_previous(self):
        if self.model.Status == gb.GRB.INFEASIBLE:
            return
        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo
        zones = self.data.zoneorder
        edges = self.data.edgeorder

        # Generation limits
        for t in taus:
            for g in generators:
                self.variables.bc_gprod_da_up[g, t].start = self.variables.bc_gprod_da_up[g, t].x
                self.variables.bc_gprod_da_down[g, t].start = self.variables.bc_gprod_da_down[g, t].x

        # Renewables and load spilled in node at time t
        for t in taus:
            for z in zones:
                self.variables.bc_winduse_da_up[z, t].start = self.variables.bc_winduse_da_up[z, t].x
                self.variables.bc_winduse_da_down[z, t].start = self.variables.bc_winduse_da_down[z, t].x
                self.variables.bc_loadshed_da_up[z, t].start = self.variables.bc_loadshed_da_up[z, t].x
                self.variables.bc_loadshed_da_down[z, t].start = self.variables.bc_loadshed_da_down[z, t].x

        # Power flow on edge e
        for e in edges:
            for t in taus:
                self.variables.bc_edgeflow_da_up[e, t].start = self.variables.bc_edgeflow_da_up[e, t].x
                self.variables.bc_edgeflow_da_down[e, t].start = self.variables.bc_edgeflow_da_down[e, t].x

    def _start_from_zero(self):

        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo
        zones = self.data.zoneorder
        edges = self.data.edgeorder

        # Generation limits
        for t in taus:
            for g in generators:
                self.variables.bc_gprod_da_up[g, t].start = 0
                self.variables.bc_gprod_da_down[g, t].start = 1

        # Renewables and load spilled in node at time t
        for t in taus:
            for z in zones:
                self.variables.bc_winduse_da_up[z, t].start = gb.GRB.UNDEFINED
                self.variables.bc_winduse_da_down[z, t].start = gb.GRB.UNDEFINED
                self.variables.bc_loadshed_da_up[z, t].start = gb.GRB.UNDEFINED
                self.variables.bc_loadshed_da_down[z, t].start = gb.GRB.UNDEFINED

        # Power flow on edge e
        for e in edges:
            for t in taus:
                self.variables.bc_edgeflow_da_up[e, t].start = gb.GRB.UNDEFINED
                self.variables.bc_edgeflow_da_down[e, t].start = gb.GRB.UNDEFINED

    def _fixed_to_zero(self):

        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo
        zones = self.data.zoneorder
        edges = self.data.edgeorder

        # Generation limits
        for t in taus:
            for g in generators:
                self.variables.bc_gprod_da_up[g, t].ub = 1
                self.variables.bc_gprod_da_up[g, t].lb = 0
                self.variables.bc_gprod_da_down[g, t].ub = 1
                self.variables.bc_gprod_da_down[g, t].lb = 0

        # Renewables and load spilled in node at time t
        for t in taus:
            for z in zones:
                self.variables.bc_winduse_da_up[z, t].ub = 1
                self.variables.bc_winduse_da_up[z, t].lb = 0
                self.variables.bc_winduse_da_down[z, t].ub = 1
                self.variables.bc_winduse_da_down[z, t].lb = 0
                self.variables.bc_loadshed_da_up[z, t].ub = 1
                self.variables.bc_loadshed_da_up[z, t].lb = 0
                self.variables.bc_loadshed_da_down[z, t].ub = 1
                self.variables.bc_loadshed_da_down[z, t].lb = 0

        # Power flow on edge e
        for e in edges:
            for t in taus:
                self.variables.bc_edgeflow_da_up[e, t].ub = 1
                self.variables.bc_edgeflow_da_up[e, t].lb = 0
                self.variables.bc_edgeflow_da_down[e, t].ub = 1
                self.variables.bc_edgeflow_da_down[e, t].lb = 0

    def _add_single_ATC_constraints(self):
        m = self.model
        taus = self.data.taus
        edges = self.data.edgeorder

        if len(taus) <= 1:
            pass

        self.constraints.single_ATC = {}
        for e in edges:
            for t1, t2 in zip(taus[:-1], taus[1:]):
                self.constraints.single_ATC[e, t1, t2] = m.addConstr(
                    self.variables.ATC[e, t1],
                    gb.GRB.EQUAL,
                    self.variables.ATC[e, t2])

    def _solve_with_zero_ATCs(self):
        '''
            Solve for optimal ATCs when all ATCs = 0.
            Useful for generating an initial feasible solution.
        '''
        atc_ubs = {}
        for k, v in self.variables.ATC.iteritems():
            atc_ubs[k] = v.ub
            v.ub = 0
        self.model.optimize()
        # In case we find that some nodes are unable to balance themselves,
        # open up all ATCs.
        if self.model.Status == gb.GRB.INFEASIBLE:
            for k, v in self.variables.ATC.iteritems():
                v.ub = 100000
                v.lb = 100000
            self.model.optimize()
        for k, v in self.variables.ATC.iteritems():
            v.ub = atc_ubs[k]
            v.lb = 0

    # def _lower_bound_from_current(self):
    #     '''
    #         When iterating Benders' cuts, we know that we can never do worse than any previous lower bound when solving the MP
    #     '''
    #     self.variables.objvar.lb = self.model.ObjBound
