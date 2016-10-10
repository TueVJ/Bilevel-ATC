import numpy as np
import gurobipy as gb
import networkx as nx
import defaults
import pickle
from myhelpers import symmetrize_dict
from itertools import izip
from collections import defaultdict
from load_fnct import load_network, load_generators, load_hvdc_links

####
#  Class to do the nodal day-ahead dispatch.
#  Init: Load network, load initial data, build model.
#  Optimize: Optimize the model.
#  Load_new_data: Takes new blocks of wind, solar and load data as input.
#                   Inserts them into the model.
####


# Class which can have attributes set.
class expando(object):
    pass


# Optimization class
class DA_Nodal:
    '''
        initial_(wind,load,solar) are (N,t) arrays where
        N is the number of nodes in the network, and
        t is the number of timesteps to optimize over.
        Note, that t is fixed by this inital assignment
    '''
    def __init__(self, initial_wind, initial_solar, initial_load):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.data = expando()
        self._load_data(initial_wind, initial_solar, initial_load)
        self._build_model()

    def optimize(self):
        self.model.optimize()

    def load_new_data(self, wind, solar, load):
        self._add_new_data(wind, solar, load)
        self._update_constraints()

    ###
    #   Loading functions
    ###

    def _load_data(self, initial_wind, initial_solar, initial_load):
        self._load_network()
        self._load_generator_data()
        self._load_intial_data(initial_wind, initial_solar, initial_load)

    def _load_network(self):
        self.data.G = load_network(defaults.nodefile, defaults.linefile)
        # # Node and edge ordering
        self.data.nodeorder = np.asarray(np.load(defaults.nodeorder_file))
        self.data.lineorder = [tuple(x) for x in np.load(defaults.lineorder_file)]
        # # Line limits
        self.data.linelimit = nx.get_edge_attributes(self.data.G, 'limit')
        zero_to_inf = lambda x: np.where(x > 0.0001, x, gb.GRB.INFINITY)
        self.data.linelimit = {k: zero_to_inf(v) for k, v in self.data.linelimit.iteritems()}
        self.data.linelimit = symmetrize_dict(self.data.linelimit)
        self.data.lineadmittance = nx.get_edge_attributes(self.data.G, 'Y')
        self.data.hvdcinfo = load_hvdc_links(defaults.hvdcfile)
        self.data.hvdcorder = [tuple(x) for x in np.load(defaults.hvdcorder_file)]
        self.data.hvdclimit = symmetrize_dict({(dat['fromNode'], dat['toNode']): zero_to_inf(dat['limit']) for i, dat in self.data.hvdcinfo.iterrows()})

    def _load_generator_data(self):
        self.data.generatorinfo = load_generators(defaults.generatorfile)
        self.data.generators = np.load(defaults.generatororder_file)
        self.data.generatorsfornode = defaultdict(list)
        origodict = self.data.generatorinfo['origin']
        for gen, n in origodict.iteritems():
            self.data.generatorsfornode[n].append(gen)

    def _load_intial_data(self, initial_wind, initial_solar, initial_load):
        if not np.shape(initial_wind) == np.shape(initial_solar) and np.shape(initial_solar) == np.shape(initial_load):
            raise ValueError("Inconsistent array shapes: wind {0}, solar {1}, load {2}".format(np.shape(initial_wind), np.shape(initial_solar), np.shape(initial_load)))

        self.data.taus = np.arange(len(initial_wind.index))
        self.data.times = initial_wind.index
        self.data.wind = initial_wind.set_index(self.data.taus)
        self.data.solar = initial_solar.set_index(self.data.taus)
        self.data.load = initial_load.set_index(self.data.taus)

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
        gendata = self.data.generatorinfo.T.to_dict()
        nodes = self.data.nodeorder
        edges = self.data.lineorder
        hvdclinks = self.data.hvdcorder
        solar = self.data.solar.to_dict()
        wind = self.data.wind.to_dict()
        load = self.data.load.to_dict()

        m = self.model

        # Production of generator g at time t
        gprod = {}
        for t in taus:
            for g in generators:
                gprod[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
        self.variables.gprod = gprod

        # Renewables and load spilled in node at time t
        renewused, loadshed = {}, {}
        for t in taus:
            for n in nodes:
                renewused[n, t] = m.addVar(lb=0.0, ub=solar[n][t] + wind[n][t])
                loadshed[n, t] = m.addVar(lb=0.0, ub=load[n][t])
        self.variables.renewused = renewused
        self.variables.loadshed = loadshed

        # Total production in node n at time t
        nodeinjection = {}
        for n in nodes:
            for t in taus:
                nodeinjection[n, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.nodeinjection = nodeinjection

        # Voltage angle at node n for time t
        voltageangle = {}
        for n in nodes:
            for t in taus:
                voltageangle[n, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.voltageangle = voltageangle

        # Power flow on edge e
        edgeflow = {}
        for e in edges:
            for t in taus:
                edgeflow[e, t] = m.addVar(lb=-self.data.linelimit[e], ub=self.data.linelimit[e])
        self.variables.edgeflow = edgeflow

        # Power flow on hvdc link dc
        hvdcflow = {}
        for dc in hvdclinks:
            for t in taus:
                hvdcflow[dc, t] = m.addVar(lb=-self.data.hvdclimit[dc], ub=self.data.hvdclimit[dc])
        self.variables.hvdcflow = hvdcflow

        m.update()

        # Slack bus setting
        for t in taus:
            voltageangle[nodes[0], t].lb = 0.0
            voltageangle[nodes[0], t].ub = 0.0

    def _build_objective(self):
        taus = self.data.taus
        nodes = self.data.nodeorder
        gendata = self.data.generatorinfo.T.to_dict()

        m = self.model
        m.setObjective(
            gb.quicksum(gendata[gen]['lincost']*self.variables.gprod[gen, t] for gen in self.data.generators for t in taus) +
            gb.quicksum(self.variables.renewused[n, t]*defaults.renew_price + self.variables.loadshed[n, t]*defaults.VOLL for n in nodes for t in taus),
            gb.GRB.MINIMIZE)

    def _build_constraints(self):
        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo.T.to_dict()
        nodes = self.data.nodeorder
        edges = self.data.lineorder
        hvdclinks = self.data.hvdcorder
        solar = self.data.solar.to_dict()
        wind = self.data.wind.to_dict()
        load = self.data.load.to_dict()

        m = self.model
        voltageangle, nodeinjection, edgeflow = self.variables.voltageangle, self.variables.nodeinjection, self.variables.edgeflow
        loadshed, renewused, gprod = self.variables.loadshed, self.variables.renewused, self.variables.gprod
        hvdcflow = self.variables.hvdcflow

        injection_to_flow = {}
        for t in taus:
            for n in nodes:
                injection_to_flow[n, t] = m.addConstr(
                    nodeinjection[n, t],
                    gb.GRB.EQUAL,
                    gb.quicksum(edgeflow[e, t] for e in edges if e[0] == n)
                    - gb.quicksum(edgeflow[e, t] for e in edges if e[1] == n)
                    + gb.quicksum(hvdcflow[dc, t] for dc in hvdclinks if dc[0] == n)
                    - gb.quicksum(hvdcflow[dc, t] for dc in hvdclinks if dc[1] == n),
                    name="CON: injection to flow at node {0}, t={1}".format(n, t))
        self.constraints.injection_to_flow = injection_to_flow

        flowangle = {}
        for t in taus:
            for e in edges:
                flowangle[e, t] = m.addConstr(
                    edgeflow[e, t],
                    gb.GRB.EQUAL,
                    self.data.lineadmittance[e] * (voltageangle[e[0], t] - voltageangle[e[1], t]))
        self.constraints.flowangle = flowangle

        powerbalance = {}
        for n in nodes:
            for t in taus:
                powerbalance[n, t] = m.addConstr(loadshed[n, t] + renewused[n, t]
                                                 + gb.quicksum(gprod[gen, t] for gen in self.data.generatorsfornode[n])
                                                 - nodeinjection[n, t],
                                                 gb.GRB.EQUAL,
                                                 load[n][t])
        self.constraints.powerbalance = powerbalance

        # systembalance = {}
        # for t in taus:
        #     systembalance[t] = m.addConstr(
        #         gb.quicksum(nodeinjection[n, t] for n in nodes),
        #         gb.GRB.EQUAL,
        #         0.0,
        #         name='System balance at time {:.00f}'.format(t))
        # self.constraints.systembalance = systembalance

    def export_schedule(self):
        taus = self.data.taus
        generators = self.data.generators
        self.results.schedule = np.array([[self.variables.gprod[g, t].x for t in taus] for g in generators])
        return self.results.schedule

    ###
    #   Data updating
    ###
    def _add_new_data(self, wind, solar, load):
        if not np.shape(wind) == np.shape(self.data.wind) and \
                np.shape(solar) == np.shape(self.data.solar) and \
                np.shape(load) == np.shape(self.data.load):
            raise ValueError("Array shape mismatch: check self.data.(wind,solar,load).shape")
        self.data.times = wind.index
        self.data.wind = wind.set_index(self.data.taus)
        self.data.solar = solar.set_index(self.data.taus)
        self.data.load = load.set_index(self.data.taus)

    def _update_constraints(self):
        taus = self.data.taus
        nodes = self.data.nodeorder
        solar = self.data.solar.to_dict()
        wind = self.data.wind.to_dict()
        load = self.data.load.to_dict()
        renewused = self.variables.renewused
        loadshed = self.variables.loadshed
        powerbalance = self.constraints.powerbalance

        for n in nodes:
            for t in taus:
                renewused[n, t].ub = solar[n][t] + wind[n][t]
                loadshed[n, t].ub = load[n][t]

        for n in nodes:
            for t in taus:
                powerbalance[n, t].rhs = load[n][t]
