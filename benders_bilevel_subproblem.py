# Import Gurobi Library
import gurobipy as gb
import numpy as np
import defaults

####
#   Benders decomposition via Gurobi + Python
####


# Class which can have attributes set.
class expando(object):
    pass


# Optimization class
class Benders_Subproblem:
    '''
        initial_(wind,load) are N [(N,t)] arrays where
        N is the number of nodes in the network, [and
        t is the number of timesteps to optimize over.
        Note, that t is fixed by this inital assignment]
    '''
    def __init__(self, MP, scenario=0):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.MP = MP
        self.data.scenario = scenario
        self.data.slackbus = self.MP.data.nodeorder[0]
        self._build_model()
        self.update_fixed_vars(MP)

    def optimize(self):
        self.model.optimize()

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model

        taus = self.MP.data.taus
        generators = self.MP.data.generators
        gendata = self.MP.data.generatorinfo
        nodes = self.MP.data.nodeorder
        lines = self.MP.data.lineorder
        wind_n_rt = self.MP.data.wind_n_rt
        load_n_rt = self.MP.data.load_n_rt
        sc = self.data.scenario

        m = self.model

        ###
        #  Primal variables
        ###

        # Production of generator g at time t, up and downregulation
        self.variables.gprod_da = {}
        self.variables.gprod_rt = {}
        self.variables.gprod_rt_up = {}
        self.variables.gprod_rt_down = {}
        for t in taus:
            for g in generators:
                self.variables.gprod_da[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                self.variables.gprod_rt[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                self.variables.gprod_rt_up[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
                self.variables.gprod_rt_down[g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])

        # Renewables and load spilled in node n at time t
        self.variables.winduse_rt, self.variables.loadshed_rt = {}, {}
        for t in taus:
            for n in nodes:
                # !
                self.variables.winduse_rt[n, t] = m.addVar(lb=0.0, ub=wind_n_rt[n][sc])
                self.variables.loadshed_rt[n, t] = m.addVar(lb=0.0, ub=load_n_rt[n])

        # Power flow on line l
        self.variables.lineflow_rt = {}
        for l in lines:
            for t in taus:
                ll = np.where(self.MP.data.linelimit[l] > 0, self.MP.data.linelimit[l], gb.GRB.INFINITY)
                self.variables.lineflow_rt[l, t] = m.addVar(lb=-ll, ub=ll)

        # Voltage angles
        self.variables.nodeangle = {}
        for t in taus:
            for n in nodes:
                if n == self.data.slackbus:
                    self.variables.nodeangle[n, t] = m.addVar(lb=0, ub=0)
                else:
                    self.variables.nodeangle[n, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)

        m.update()

    def _build_objective(self):
        m = self.model

        taus = self.MP.data.taus
        generators = self.MP.data.generators
        gendata = self.MP.data.generatorinfo
        nodes = self.MP.data.nodeorder
        lines = self.MP.data.lineorder
        wind_n_rt = self.MP.data.wind_n_rt
        load_n_rt = self.MP.data.load_n_rt

        self.model.setObjective(
            gb.quicksum(
                (gendata[g]['lincost'] + defaults.up_redispatch_premium)*self.variables.gprod_rt_up[g, t] +
                (-gendata[g]['lincost'] + defaults.down_redispatch_premium)*self.variables.gprod_rt_down[g, t]
                for g in generators for t in taus) +
            gb.quicksum(
                self.variables.winduse_rt[n, t]*defaults.renew_price +
                self.variables.loadshed_rt[n, t]*defaults.VOLL
                for n in nodes for t in taus),
            gb.GRB.MINIMIZE)

    def _build_constraints(self):
        m = self.model

        taus = self.MP.data.taus
        generators = self.MP.data.generators
        gendata = self.MP.data.generatorinfo
        nodes = self.MP.data.nodeorder
        lines = self.MP.data.lineorder
        wind_n_rt = self.MP.data.wind_n_rt
        load_n_rt = self.MP.data.load_n_rt
        sc = self.data.scenario

        # RT power balance
        self.constraints.powerbalance_rt = {}
        for n in nodes:
            for t in taus:
                # !
                self.constraints.powerbalance_rt[n, t] = m.addConstr(
                    gb.quicksum(self.variables.gprod_rt[gen, t] for gen in self.MP.data.node_to_generators[n])
                    + self.variables.loadshed_rt[n, t] + self.variables.winduse_rt[n, t]
                    - gb.quicksum(self.variables.lineflow_rt[l, t] for l in lines if l[0] == n)
                    + gb.quicksum(self.variables.lineflow_rt[l, t] for l in lines if l[1] == n),
                    gb.GRB.EQUAL,
                    self.MP.data.load_n_rt[n])
                # self.data.load_n_rt[n, t]) # !!!

        # Up and down-regulation
        self.constraints.gprod_regulation = {}
        for t in taus:
            for g in generators:
                self.constraints.gprod_regulation[g, t] = m.addConstr(
                    self.variables.gprod_rt[g, t],
                    gb.GRB.EQUAL,
                    self.variables.gprod_da[g, t]
                    + self.variables.gprod_rt_up[g, t]
                    - self.variables.gprod_rt_down[g, t])

        # DC flow constraint
        self.constraints.flow_to_angle = {}
        for t in taus:
            for l in lines:
                self.constraints.flow_to_angle[l, t] = m.addConstr(
                    self.variables.lineflow_rt[l, t],
                    gb.GRB.EQUAL,
                    (self.variables.nodeangle[l[0], t] - self.variables.nodeangle[l[1], t])*self.MP.data.lineadmittance[l])

        # Constrain day-ahead production
        self.constraints.fix_da_production = {}
        for t in taus:
            for g in generators:
                self.constraints.fix_da_production[g, t] = m.addConstr(
                    self.variables.gprod_da[g, t],
                    gb.GRB.EQUAL,
                    self.MP.variables.gprod_da[g, t].x)

    def update_fixed_vars(self, MP):
        # p_DA update

        taus = self.MP.data.taus
        generators = self.MP.data.generators

        for t in taus:
            for g in generators:
                self.constraints.fix_da_production[g, t].rhs = MP.variables.gprod_da[g, t].x
        pass
