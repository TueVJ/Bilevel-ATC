import numpy as np
import gurobipy as gb
import networkx as nx
import defaults
import pickle
from myhelpers import symmetrize_dict, invert_dict
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
class DA_Zonal:
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
        self.data.linelimit = symmetrize_dict(self.data.linelimit)
        # # HVDC links
        self.data.hvdcinfo = load_hvdc_links(defaults.hvdcfile)
        # self.data.hvdcorder = [tuple(x) for x in np.load(defaults.hvdcorder_file)]
        self.data.hvdcorder = []
        self.data.hvdclimit = symmetrize_dict({(dat['fromNode'], dat['toNode']): dat['limit'] for i, dat in self.data.hvdcinfo.iterrows()})
        # # Load Zones
        self.data.zoneorder = np.asarray(np.load(defaults.zoneorder_file))
        self.data.zoneedgeorder = [tuple(x) for x in np.load(defaults.zoneedgeorder_file)]
        self.data.node_to_zone = nx.get_node_attributes(self.data.G, 'country')
        self.data.zone_to_nodes = invert_dict(self.data.node_to_zone)
        # # Construct edge limit dictionary (= sum of underlying capacity)
        self.data.edgelimit = defaultdict(lambda: 0)
        for e in self.data.lineorder:
            z1 = self.data.node_to_zone[e[0]]
            z2 = self.data.node_to_zone[e[1]]
            if z1 != z2:
                if (z1, z2) in self.data.zoneedgeorder:
                    self.data.edgelimit[z1, z2] += self.data.linelimit[e[0], e[1]]
                elif (z2, z1) in self.data.zoneedgeorder:
                    self.data.edgelimit[z2, z1] += self.data.linelimit[e[1], e[0]]
                else:
                    raise ValueError('Line not found: {}'.format(e))
        for dc in self.data.hvdcorder:
            z1 = self.data.node_to_zone[dc[0]]
            z2 = self.data.node_to_zone[dc[1]]
            if z1 != z2:
                if (z1, z2) in self.data.zoneedgeorder:
                    self.data.edgelimit[z1, z2] += self.data.hvdclimit[dc[0], dc[1]]
                elif (z2, z1) in self.data.zoneedgeorder:
                    self.data.edgelimit[z2, z1] += self.data.hvdclimit[dc[1], dc[0]]
                else:
                    raise ValueError('HVDC link not found: {}'.format(e))
        # Interpret zonal edges with 0 capacity as having unlimited capacity
        zero_to_inf = lambda x: np.where(x > 0.0001, x, gb.GRB.INFINITY)
        self.data.zoneedgelimit = {k: zero_to_inf(v) for k, v in self.data.edgelimit.iteritems()}
        self.data.zoneedgelimit[('ITA', 'GRC')] = 0
        self.data.zoneedgelimit[('GRC', 'ITA')] = 0

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
        zones = self.data.zoneorder
        zoneedges = self.data.zoneedgeorder
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
            for z in zones:
                renewused[z, t] = m.addVar(lb=0.0, ub=sum(solar[n][t] + wind[n][t] for n in self.data.zone_to_nodes[z]))
                loadshed[z, t] = m.addVar(lb=0.0, ub=sum(load[n][t] for n in self.data.zone_to_nodes[z]))
        self.variables.renewused = renewused
        self.variables.loadshed = loadshed

        # Total production in node n at time t
        nodeinjection = {}
        for z in zones:
            for t in taus:
                nodeinjection[z, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.nodeinjection = nodeinjection

        # Power flow on edge e
        edgeflow = {}
        for e in zoneedges:
            for t in taus:
                edgeflow[e, t] = m.addVar(lb=-self.data.zoneedgelimit[e], ub=self.data.zoneedgelimit[e])
        self.variables.edgeflow = edgeflow

        m.update()

    def _build_objective(self):
        taus = self.data.taus
        zones = self.data.zoneorder
        gendata = self.data.generatorinfo.T.to_dict()

        m = self.model
        m.setObjective(
            gb.quicksum(gendata[gen]['lincost']*self.variables.gprod[gen, t] for gen in self.data.generators for t in taus) +
            gb.quicksum(self.variables.renewused[z, t]*defaults.renew_price + self.variables.loadshed[z, t]*defaults.VOLL for z in zones for t in taus),
            gb.GRB.MINIMIZE)

    def _build_constraints(self):
        taus = self.data.taus
        generators = self.data.generators
        gendata = self.data.generatorinfo.T.to_dict()
        zones = self.data.zoneorder
        zoneedges = self.data.zoneedgeorder
        solar = self.data.solar.to_dict()
        wind = self.data.wind.to_dict()
        load = self.data.load.to_dict()

        m = self.model
        nodeinjection, edgeflow = self.variables.nodeinjection, self.variables.edgeflow
        loadshed, renewused, gprod = self.variables.loadshed, self.variables.renewused, self.variables.gprod

        injection_to_flow = {}
        for t in taus:
            for z in zones:
                injection_to_flow[z, t] = m.addConstr(
                    nodeinjection[z, t],
                    gb.GRB.EQUAL,
                    gb.quicksum(edgeflow[e, t] for e in zoneedges if e[0] == z)
                    - gb.quicksum(edgeflow[e, t] for e in zoneedges if e[1] == z),
                    name="CON: injection to flow at zone {0}, t={1}".format(z, t))
        self.constraints.injection_to_flow = injection_to_flow

        powerbalance = {}
        for z in zones:
            for t in taus:
                powerbalance[z, t] = m.addConstr(
                    loadshed[z, t] + renewused[z, t]
                    + gb.quicksum(gprod[gen, t] for n in self.data.zone_to_nodes[z] for gen in self.data.generatorsfornode[n])
                    - nodeinjection[z, t],
                    gb.GRB.EQUAL,
                    sum(load[n][t] for n in self.data.zone_to_nodes[z]))
        self.constraints.powerbalance = powerbalance

        systembalance = {}
        for t in taus:
            systembalance[t] = m.addConstr(
                gb.quicksum(nodeinjection[z, t] for z in zones),
                gb.GRB.EQUAL,
                0.0,
                name='System balance at time {:.00f}'.format(t))
        self.constraints.systembalance = systembalance

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
        zones = self.data.zoneorder
        solar = self.data.solar.to_dict()
        wind = self.data.wind.to_dict()
        load = self.data.load.to_dict()
        renewused = self.variables.renewused
        loadshed = self.variables.loadshed
        powerbalance = self.constraints.powerbalance

        for t in taus:
            for z in zones:
                renewused[z, t].ub = sum(solar[n][t] + wind[n][t] for n in self.data.zone_to_nodes[z])
                loadshed[z, t].ub = sum(load[n][t] for n in self.data.zone_to_nodes[z])

        for z in zones:
            for t in taus:
                powerbalance[z, t].rhs = sum(load[n][t] for n in self.data.zone_to_nodes[z])
