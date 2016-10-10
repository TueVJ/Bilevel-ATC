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
class Smeers_ATC_Maximization_nodes:
    def __init__(self, z1, z2):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self._load_data(z1, z2)
        self._build_model()

    def optimize(self):
        self.model.optimize()

    ###
    #   Loading functions
    ###

    def _load_data(self, n1, n2):
        self.data.n1 = n1
        self.data.n2 = n2
        self._load_network()
        self.data.bigM = 10*max(self.data.linelimit.itervalues())

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
        nodes = self.data.nodeorder
        edges = self.data.lineorder
        hvdclinks = self.data.hvdcorder
        n1 = self.data.n1
        n2 = self.data.n2

        m = self.model

        # Injection into n1, extraction from n2
        self.variables.totexchange = m.addVar(lb=0, ub=gb.GRB.INFINITY)

        # Total production in node n
        nodeinjection = {}
        for n in nodes:
            nodeinjection[n] = m.addVar(lb=0, ub=0)
        self.variables.nodeinjection = nodeinjection

        # Voltage angle at node n
        voltageangle = {}
        for n in nodes:
            voltageangle[n] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.voltageangle = voltageangle

        # Power flow on edge e
        edgeflow = {}
        for e in edges:
            edgeflow[e] = m.addVar(lb=-self.data.linelimit[e], ub=self.data.linelimit[e])
        self.variables.edgeflow = edgeflow

        # Power flow on hvdc link dc
        hvdcflow = {}
        for dc in hvdclinks:
            hvdcflow[dc] = m.addVar(lb=-self.data.hvdclimit[dc], ub=self.data.hvdclimit[dc])
        self.variables.hvdcflow = hvdcflow

        m.update()

        # Slack bus setting
        voltageangle[nodes[0]].lb = 0.0
        voltageangle[nodes[0]].ub = 0.0

    def _build_objective(self):
        m = self.model
        m.setObjective(self.variables.totexchange, gb.GRB.MAXIMIZE)

    def _build_constraints(self):
        nodes = self.data.nodeorder
        edges = self.data.lineorder
        hvdclinks = self.data.hvdcorder
        n1 = self.data.n1
        n2 = self.data.n2

        nodeinjection = self.variables.nodeinjection
        totexchange = self.variables.totexchange
        edgeflow = self.variables.edgeflow

        m = self.model
        voltageangle, nodeinjection, edgeflow = self.variables.voltageangle, self.variables.nodeinjection, self.variables.edgeflow
        hvdcflow = self.variables.hvdcflow

        injection_to_flow = {}
        for n in nodes:
            injection_to_flow[n] = m.addConstr(
                nodeinjection[n],
                gb.GRB.EQUAL,
                gb.quicksum(edgeflow[e] for e in edges if e[0] == n)
                - gb.quicksum(edgeflow[e] for e in edges if e[1] == n)
                + gb.quicksum(hvdcflow[dc] for dc in hvdclinks if dc[0] == n)
                - gb.quicksum(hvdcflow[dc] for dc in hvdclinks if dc[1] == n),
                name="CON: injection to flow at node {0}".format(n))
        self.constraints.injection_to_flow = injection_to_flow

        flowangle = {}
        for e in edges:
            flowangle[e] = m.addConstr(
                edgeflow[e],
                gb.GRB.EQUAL,
                self.data.lineadmittance[e] * (voltageangle[e[0]] - voltageangle[e[1]]))
        self.constraints.flowangle = flowangle

        # Constrain totexchange by
        # injection into zone 1, extraction from zone 2
        self.constraints.injection_coupling_in = m.addConstr(
            totexchange, gb.GRB.EQUAL, nodeinjection[n1])
        self.constraints.injection_coupling_out = m.addConstr(
            -totexchange, gb.GRB.EQUAL, nodeinjection[n2])
        nodeinjection[n1].ub = gb.GRB.INFINITY
        nodeinjection[n2].lb = -gb.GRB.INFINITY

    def update_injection_nodes(self, n1, n2):
        nodeinjection = self.variables.nodeinjection
        totexchange = self.variables.totexchange
        m = self.model

        self.model.remove(self.constraints.injection_coupling_in)
        self.model.remove(self.constraints.injection_coupling_out)

        self.constraints.injection_coupling_in = m.addConstr(
            totexchange, gb.GRB.EQUAL, nodeinjection[n1])
        self.constraints.injection_coupling_out = m.addConstr(
            totexchange, gb.GRB.EQUAL, -1.0 * nodeinjection[n2])
        m.update()

        n1old = self.data.n1
        n2old = self.data.n2
        nodeinjection[n1old].ub = 0
        nodeinjection[n2old].lb = 0
        nodeinjection[n1].ub = gb.GRB.INFINITY
        nodeinjection[n2].lb = -gb.GRB.INFINITY
        m.update()

        self.data.n1 = n1
        self.data.n2 = n2
