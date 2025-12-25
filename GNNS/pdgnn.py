import numpy as np
import pandas as pd 
import pickle
from pypower import case6ww, case14, idx_bus, idx_gen, idx_brch
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
import torch.nn.functional as F
import sys
import os
from GNNS.GNN_classifier import PDFeasNetworkV2, PDFeasNetworkV3
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from GNNS.opf_functions import *
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from math import sin, cos

name_to_func = {"dp_dp": dp_dp,
                "dq_dq": dq_dq,
                "dp_dv": dp_dv,
                "dp_dtheta": dp_dtheta,
                "dq_dv": dq_dv,
                "dq_dtheta": dq_dtheta,
                "dpn_dv": dpn_dv,
                "dpn_dtheta": dpn_dtheta,
                "dqn_dv": dqn_dv,
                "dqn_dtheta": dqn_dtheta,
                "pbal_lambda": pbal_lambda,
                "qbal_lambda": qbal_lambda,
                "boundary_min": boundary_min,
                "boundary_max": boundary_max,
                "boundary_lambda_vmin": boundary_lambda_vmin,
                "boundary_lambda_vmax": boundary_lambda_vmax,
                "boundary_lambda_pmin": boundary_lambda_pmin,
                "boundary_lambda_pmax": boundary_lambda_pmax,
                "pflow_lambda": pflow_lambda,
                "qflow_lambda": qflow_lambda,
                "dpflow_dvi": dpflow_dvi,
                "dqflow_dvi": dqflow_dvi,
                "dpflow_dvj": dpflow_dvj,
                "dqflow_dvj": dqflow_dvj,
                "dpflow_dthetai": dpflow_dthetai,
                "dpflow_dthetaj": dpflow_dthetaj,
                "dqflow_dthetai": dqflow_dthetai,
                "dqflow_dthetaj": dqflow_dthetaj,
                "dpflow_dp": dpflow_dp,
                "dqflow_dq": dqflow_dq,
                "apparent_flow_lambda": apparent_flow_lambda,
                "dapparent_flow": dapparent_flow,
                "variable_objective": variable_objective}

class PDGnn(): 
    def __init__(self, test_case): 
        self.test_case = test_case
        # create variables 
        self.create_variables()
        # create constraints and objective function 
        self.create_problem_constraints()
        self.feas_f = self.f_node.copy() # copy edges for feasibility that don't include objective
        self.feas_t = self.t_node.copy()
        self.create_objective_function()
        # store edge indices 
        self.feas_index = EdgeIndex([self.feas_f, self.feas_t])
        self.opt_index = EdgeIndex([self.f_node, self.t_node])
        # store data 
        self.dataset = None

    def create_boundary_constraints(self, var_ind, constr_ind, var_helper, constr_helper, var_deriv, constr_deriv): 
        # update lookup tables
        self.deriv_lookup[var_ind][constr_ind] = var_deriv  
        self.deriv_lookup[constr_ind][var_ind] = constr_deriv  
        self.lookup_helper[var_ind][constr_ind] = var_helper
        self.lookup_helper[constr_ind] = constr_helper
        # update edge information
        self.f_node += [var_ind, constr_ind]
        self.t_node += [constr_ind, var_ind]
        return 
    
    def create_bus_equality_constraints(self, bus_ind, constr_ind, p = True): 
        # obtain indices for the bus 
        vm_ind, va_ind = self.var_dict[f'bus_{bus_ind}'][0:2]
        pg_inds, qg_inds = [], []
        for i, ind in enumerate(self.var_dict[f'bus_{bus_ind}'][2:]):
            if i % 2 == 0: 
                pg_inds.append(ind)
            else: 
                qg_inds.append(ind)
        pd_ind, qd_ind = self.var_dict[f'busload_{bus_ind}'][0:2]
        # store information for every neighbor
        neighbor_helpers = [[vm_ind, va_ind, 0, pg_inds, pd_ind, qg_inds, qd_ind]] # TODO: check self admittance!!
        branch_inds = [i for i in range(len(self.test_case['branch'])) if bus_ind in [self.test_case['branch'][i][idx_brch.F_BUS], self.test_case['branch'][i][idx_brch.T_BUS]]]
        for bi in branch_inds: 
            branch = self.test_case['branch'][bi]
            nbus = int(branch[idx_brch.F_BUS]) if int(branch[idx_brch.T_BUS]) == bus_ind else int(branch[idx_brch.T_BUS])
            # get neighbor branch information and line information 
            nvm, nva = self.var_dict[f'bus_{nbus}'][0:2]
            real_ad = branch[idx_brch.BR_R]/(branch[idx_brch.BR_R]**2 + branch[idx_brch.BR_X]**2)
            im_ad = -1*branch[idx_brch.BR_X]/(branch[idx_brch.BR_R]**2 + branch[idx_brch.BR_X]**2)   
            n_info = [nvm, nva, real_ad, im_ad]
            neighbor_helpers.append(n_info)
            # create connection for neighboring vm and va 
            self.deriv_lookup[nvm][constr_ind] = dpn_dv if p else dqn_dv
            self.lookup_helper[nvm][constr_ind] = n_info 
            self.f_node.append(nvm)
            self.t_node.append(constr_ind)
            self.deriv_lookup[nva][constr_ind] = dpn_dtheta if p else dqn_dtheta
            self.lookup_helper[nva][constr_ind] = n_info 
            self.f_node.append(nva)
            self.t_node.append(constr_ind)
        # with all neighbor information, store connections with vm
        self.deriv_lookup[vm_ind][constr_ind] = dp_dv if p else dq_dv 
        self.lookup_helper[vm_ind][constr_ind] = neighbor_helpers
        self.f_node.append(vm_ind)
        self.t_node.append(constr_ind)
        # store connections with va
        self.deriv_lookup[va_ind][constr_ind] = dp_dtheta if p else dq_dtheta 
        self.lookup_helper[va_ind][constr_ind] = neighbor_helpers
        self.f_node.append(va_ind)
        self.t_node.append(constr_ind)
        # store connections with real and reactive generation 
        gen_inds = pg_inds if p else qg_inds 
        for gen_ind in gen_inds: 
            self.deriv_lookup[gen_ind][constr_ind] = dp_dp if p else dq_dq
            self.f_node.append(gen_ind)
            self.t_node.append(constr_ind)
        # store connection with constraint itself, from bus voltage to constraint 
        self.deriv_lookup[constr_ind][vm_ind] = pbal_lambda if p else qbal_lambda
        self.lookup_helper[constr_ind] = neighbor_helpers 
        self.f_node.append(constr_ind)
        self.t_node.append(vm_ind)
        return

    def create_line_constraints(self, line_ind, constr_ind): 
        p_constr, q_constr, s_constr = constr_ind, constr_ind + 1, constr_ind + 2
        self.constr_dict[f'line_{line_ind}'] = [p_constr, q_constr, s_constr]
        self.deriv_lookup[p_constr] = {}
        self.deriv_lookup[q_constr] = {}
        self.deriv_lookup[s_constr] = {}
        branch = self.test_case['branch'][line_ind]
        base_mva = self.test_case['baseMVA']
        # obtain indices for the line 
        vm_i, va_i = self.var_dict[f'bus_{int(branch[idx_brch.F_BUS])}'][0:2]
        vm_j, va_j = self.var_dict[f'bus_{int(branch[idx_brch.T_BUS])}'][0:2]
        pflow, qflow = self.var_dict[f'branch_{line_ind}']
        G = branch[idx_brch.BR_R]/(branch[idx_brch.BR_R]**2 + branch[idx_brch.BR_X]**2)
        B = -1*branch[idx_brch.BR_X]/(branch[idx_brch.BR_R]**2 + branch[idx_brch.BR_X]**2)
        tmax = branch[idx_brch.RATE_A]/base_mva
        p_helper = [[vm_i, va_i], [vm_j, va_j], pflow, [G, B]]
        q_helper = [[vm_i, va_i], [vm_j, va_j], qflow, [G, B]]
        # constraints for pflow equality
        self.deriv_lookup[p_constr][pflow] = pflow_lambda 
        self.lookup_helper[p_constr] = p_helper 
        self.deriv_lookup[pflow][p_constr] = dpflow_dp
        self.lookup_helper[pflow][p_constr] = p_helper
        self.f_node += [p_constr, pflow]
        self.t_node += [pflow, p_constr]
        # constraints for qflow equality 
        self.deriv_lookup[q_constr][qflow] = qflow_lambda 
        self.lookup_helper[q_constr] = q_helper 
        self.deriv_lookup[qflow][q_constr] = dqflow_dq
        self.lookup_helper[qflow][q_constr] = q_helper
        self.f_node += [q_constr, qflow]
        self.t_node += [qflow, q_constr]
        # constraints for pflow equality WRT vm_i, va_i, vm_j, va_j 
        self.deriv_lookup[vm_i][p_constr] = dpflow_dvi
        self.lookup_helper[vm_i][p_constr] = p_helper
        self.deriv_lookup[vm_j][p_constr] = dpflow_dvj
        self.lookup_helper[vm_j][p_constr] = p_helper
        self.deriv_lookup[va_i][p_constr] = dpflow_dthetai
        self.lookup_helper[va_i][p_constr] = p_helper
        self.deriv_lookup[va_j][p_constr] = dpflow_dthetaj
        self.lookup_helper[va_j][p_constr] = p_helper
        self.f_node += [vm_i, vm_j, va_i, va_j]
        self.t_node += [p_constr, p_constr, p_constr, p_constr]
        # constraints for qflow equality WRT vm_i, vm_j, va_i, va_j
        self.deriv_lookup[vm_i][q_constr] = dqflow_dvi
        self.lookup_helper[vm_i][q_constr] = q_helper
        self.deriv_lookup[vm_j][q_constr] = dqflow_dvj
        self.lookup_helper[vm_j][q_constr] = q_helper
        self.deriv_lookup[va_i][q_constr] = dqflow_dthetai
        self.lookup_helper[va_i][q_constr] = q_helper
        self.deriv_lookup[va_j][q_constr] = dqflow_dthetaj
        self.lookup_helper[va_j][q_constr] = q_helper
        self.f_node += [vm_i, vm_j, va_i, va_j]
        self.t_node += [q_constr, q_constr, q_constr, q_constr]
        # constraints for apparent power flow 
        self.deriv_lookup[s_constr][pflow] = apparent_flow_lambda
        self.lookup_helper[s_constr] = [pflow, qflow, tmax]
        self.deriv_lookup[pflow][s_constr] = dapparent_flow
        self.deriv_lookup[qflow][s_constr] = dapparent_flow
        self.proj_dict[s_constr] = [0, 1e6]
        self.f_node += [qflow, pflow, s_constr]
        self.t_node += [s_constr, s_constr, pflow]
        return constr_ind + 3

    def create_problem_constraints(self): 
        self.f_node, self.t_node = [], [] # hold edge information for constraints 
        self.deriv_lookup = {i: {} for i in range(self.num_vars)}
        self.lookup_helper = {i: {} for i in range(self.num_vars)} 
        self.proj_dict, self.constr_dict = {}, {}
        constr_counter = self.num_vars
        base_mva = self.test_case['baseMVA']
        # generator constraints 
        for i, gen in enumerate(self.test_case['gen']): 
            pg_ind, qg_ind = self.var_dict[f'gen_{i}'] # variable indices 
            gen_stat = self.var_dict[f'genstat_{i}'][0] # generator status
            # pg bounds (include gen status)
            pg_lb, pg_ub = constr_counter, constr_counter + 1 # constraint indices 
            self.deriv_lookup[constr_counter] = {}
            self.deriv_lookup[constr_counter + 1] = {}
            self.create_boundary_constraints(pg_ind, pg_lb, [], [gen[idx_gen.PMIN]/base_mva, gen_stat], boundary_min, boundary_lambda_pmin)
            self.create_boundary_constraints(pg_ind, pg_ub, [], [gen[idx_gen.PMAX]/base_mva, gen_stat], boundary_max, boundary_lambda_pmax)
            # inequality constraints have lower bound of 0
            self.proj_dict[pg_ind] = [gen[idx_gen.PMIN]/base_mva, gen[idx_gen.PMAX]/base_mva]
            self.proj_dict[pg_lb] = [0, 1e6]
            self.proj_dict[pg_ub] = [0, 1e6]
            constr_counter += 2 
            #  qg bounds 
            qg_lb, qg_ub = constr_counter, constr_counter + 1 # constraint indices 
            self.deriv_lookup[constr_counter] = {}
            self.deriv_lookup[constr_counter + 1] = {}
            self.create_boundary_constraints(qg_ind, qg_lb, [], [gen[idx_gen.QMIN]/base_mva, gen_stat], boundary_min, boundary_lambda_pmin)
            self.create_boundary_constraints(qg_ind, qg_ub, [], [gen[idx_gen.QMAX]/base_mva, gen_stat], boundary_max, boundary_lambda_pmax)
            self.proj_dict[qg_ind] = [gen[idx_gen.QMIN]/base_mva, gen[idx_gen.QMAX]/base_mva]
            self.constr_dict[f'gen_{i}'] = [pg_lb, pg_ub, qg_lb, qg_ub]
            self.proj_dict[qg_lb] = [0, 1e6]
            self.proj_dict[qg_ub] = [0, 1e6]
            constr_counter += 2 
        # bus constraints 
        for i, bus in enumerate(self.test_case['bus']): 
            # vm boundaries
            vm_ind = self.var_dict[f'bus_{i}'][0]
            vm_lb, vm_ub = constr_counter, constr_counter + 1 
            self.deriv_lookup[constr_counter] = {}
            self.deriv_lookup[constr_counter + 1] = {}
            self.create_boundary_constraints(vm_ind, vm_lb, [], bus[idx_bus.VMIN], boundary_min, boundary_lambda_vmin)
            self.create_boundary_constraints(vm_ind, vm_ub, [], bus[idx_bus.VMAX], boundary_max, boundary_lambda_vmax)
            self.proj_dict[vm_ind] = [bus[idx_bus.VMIN], bus[idx_bus.VMAX]]
            self.constr_dict[f'vm_{i}'] = [vm_lb, vm_ub]
            self.proj_dict[vm_lb] = [0, 1e6]
            self.proj_dict[vm_ub] = [0, 1e6]
            constr_counter += 2
            #  bus equality constraints 
            pbal, qbal = constr_counter, constr_counter + 1
            self.deriv_lookup[constr_counter] = {}
            self.deriv_lookup[constr_counter + 1] = {}
            self.create_bus_equality_constraints(i, pbal)
            self.create_bus_equality_constraints(i, qbal, p = False)
            self.constr_dict[f'bus_{i}'] = [pbal, qbal]
            constr_counter += 2 
        # branch constraints 
        for i, branch in enumerate(self.test_case['branch']): 
            constr_counter = self.create_line_constraints(i, constr_counter) 
        self.num_constrs = constr_counter - 1 - self.num_vars
        return

    def create_objective_function(self): 
        # iterate through generators 
        for g, gencost in enumerate(self.test_case["gencost"]): 
            # grab cost terms, reverse order (zero power to highest power)
            cost_terms = gencost[4:][::-1]
            helper_costs = [] # derivative of cost
            for i, term in enumerate(cost_terms): 
                helper_costs.append(i * term)
            # add edge between the generator (pg) and itself 
            pg = self.var_dict[f'gen_{g}'][0]
            self.deriv_lookup[pg][pg] = variable_objective
            self.lookup_helper[pg][pg] = helper_costs 
            self.f_node.append(pg)
            self.t_node.append(pg)

    def create_variables(self): 
        self.var_dict = {}
        var_counter = 0 
        # store generators on each bus
        gen_buses = [[] for _ in range(len(self.test_case['bus']))]
        for i, gen in enumerate(self.test_case['gen']): 
            gen_buses[int(gen[idx_gen.GEN_BUS])].append(i)
        
        # create variables
        for i in range(len(self.test_case['bus'])):
            vm, va = var_counter, var_counter + 1 # each bus gets a va and a vm
            var_counter += 2 
            bus_vars = [vm, va]
            # for each generator on the bus, the bus also gets a pg and qg.
            for gen in gen_buses[i]: 
                pg, qg = var_counter, var_counter + 1 
                var_counter += 2 
                bus_vars += [pg, qg]
                self.var_dict[f'gen_{gen}'] = [pg, qg]
            # store variables on this bus 
            self.var_dict[f'bus_{i}'] = bus_vars

        
        for i in range(len(self.test_case['branch'])): 
            pflow, qflow = var_counter, var_counter + 1 # real and reactive flow on a line 
            var_counter += 2
            self.var_dict[f'branch_{i}'] = [pflow, qflow]

        for i in range(len(self.test_case['gen'])): 
            # each generator has a status 
            stat = var_counter
            var_counter += 1 
            self.var_dict[f'genstat_{i}'] = [stat]

        for i in range(len(self.test_case['bus'])): 
            # each bus has a real and reactive load 
            pload, qload = var_counter, var_counter + 1 
            self.var_dict[f'busload_{i}'] = [pload, qload]
            var_counter += 2
        self.num_vars = var_counter
        return 
    
    def create_model(self, model_type, batch_size = 20, optim_lst = [], feas_lst = [], conv_tol = 0.01): 
        # prepare values 
        n_edges, nf_edges = len(self.f_node), len(self.feas_f)
        num_nodes = self.num_vars + self.num_constrs
        # create model object 
        model_dict = {
            'feasv2': PDFeasNetworkV2,
            'feasv3': PDFeasNetworkV3
        }
        model_class = model_dict[model_type]
        model = model_class(self.deriv_lookup, n_edges, nf_edges,
                 num_nodes, batch_size, optim_lst, feas_lst,  conv_tol, 
                 self.lookup_helper, self.num_vars, proj_dict = self.proj_dict)
        return model

    def create_ipopt_model(self, row): 
        base_mva = self.test_case['baseMVA']
        # create model 
        model = ConcreteModel()
        # create variable list 
        self.num_predicted_values = self.num_vars - 2*len(self.test_case['branch']) - 2*len(self.test_case['bus']) - len(self.test_case['gen'])
        model.x = VarList(domain=Reals)
        for i in range(self.num_predicted_values): 
            model.x.add()
        # add vm bounds 
        for i, bus in enumerate(self.test_case['bus']): 
            vm_ind = self.var_dict[f'bus_{i}'][0]
            model.x[vm_ind+1].lb = bus[idx_bus.VMIN]
            model.x[vm_ind+1].ub = bus[idx_bus.VMAX]

        # add constraints 
        constr_lst = []
        # bus equality 
        for bus_ind, bus in enumerate(self.test_case['bus']): 
            # get this bus information 
            if len(self.var_dict[f'bus_{bus_ind}']) > 2:
                vm, va, pg, qg = self.var_dict[f'bus_{bus_ind}']
                vm, va, pg, qg = model.x[vm+1], model.x[va+1], model.x[pg+1], model.x[qg+1]
            else:  
                vm, va = self.var_dict[f'bus_{bus_ind}']
                vm, va = model.x[vm+1], model.x[va+1]
                pg, qg = 0, 0
            pd, qd = row[f'pd_{bus_ind}']/base_mva, row[f'qd_{bus_ind}']/base_mva
            # get neighboring bus information
            neighbors = []
            branch_inds = [i for i in range(len(self.test_case['branch'])) if i in [self.test_case['branch'][bus_ind][idx_brch.F_BUS], self.test_case['branch'][bus_ind][idx_brch.T_BUS]]]
            for bi in branch_inds: 
                branch = self.test_case['branch'][bi]
                nbus = int(branch[idx_brch.F_BUS]) if int(branch[idx_brch.T_BUS]) == bus_ind else int(branch[idx_brch.T_BUS])
                if nbus == bus_ind: 
                    continue
                # get neighbor branch information and line information 
                nvm, nva = self.var_dict[f'bus_{nbus}'][0:2]
                nvm, nva = model.x[nvm+1], model.x[nva+1]
                real_ad = branch[idx_brch.BR_R]/(branch[idx_brch.BR_R]**2 + branch[idx_brch.BR_X]**2)
                im_ad = -1*branch[idx_brch.BR_X]/(branch[idx_brch.BR_R]**2 + branch[idx_brch.BR_X]**2)   
                n_info = [nvm, nva, real_ad, im_ad]
                neighbors.append(n_info)
            # add real and reactive power flow constraints 
            constr_lst.append(real_power_balance(vm, va, pg, pd, 0, neighbors) == 0)
            constr_lst.append(reactive_power_balance(vm, va, qg, qd, 0, neighbors) == 0)
        # generator bounds  
        for gen_ind, gen in enumerate(self.test_case['gen']):
            pg, qg = self.var_dict[f'gen_{gen_ind}']
            pg, qg = model.x[pg+1], model.x[qg+1]
            genstat = row[f'status_{gen_ind}']
            # constr_lst += upper_lower_gen_boundaries(pg, gen[idx_gen.PMIN]/base_mva,
            #                                          gen[idx_gen.PMAX]/base_mva,
            #                                          genstat)
            constr_lst += upper_lower_gen_boundaries(qg, gen[idx_gen.QMIN]/base_mva,
                                                     gen[idx_gen.QMAX]/base_mva,
                                                     genstat)
            
        # branch thermal limit 
        for i, branch in enumerate(self.test_case['branch']): 
            fbus, tbus = int(branch[idx_brch.F_BUS]), int(branch[idx_brch.T_BUS])
            G = branch[idx_brch.BR_R]/(branch[idx_brch.BR_R]**2 + branch[idx_brch.BR_X]**2)
            B = -1*branch[idx_brch.BR_X]/(branch[idx_brch.BR_R]**2 + branch[idx_brch.BR_X]**2)
            tmax = branch[idx_brch.RATE_A]/base_mva
            vm_i, va_i = model.x[self.var_dict[f'bus_{fbus}'][0]+1], model.x[self.var_dict[f'bus_{fbus}'][1]+1]
            vm_j, va_j = model.x[self.var_dict[f'bus_{tbus}'][0]+1], model.x[self.var_dict[f'bus_{tbus}'][1]+1]
            pflow = pflow_constraint(vm_i, va_i, vm_j, va_j, G, B)
            qflow = qflow_constraint(vm_i, va_i, vm_j, va_j, G, B)
            constr_lst.append(pflow**2 + qflow**2 - tmax**2 <= 0)
        # add constraints to model 
        model.constraints = ConstraintList()
        for con in constr_lst:
            model.constraints.add(con)
        # create objective 
        obj_expr = 0
        for g, gencost in enumerate(self.test_case["gencost"]): 
            gen = model.x[self.var_dict[f'gen_{g}'][0]+1]
            # grab cost terms, reverse order (zero power to highest power)
            cost_terms = gencost[4:][::-1]
            helper_costs = [] # derivative of cost
            for i, term in enumerate(cost_terms): 
                helper_costs.append(i * term)
            obj_expr += min_cost(gen, helper_costs)
        # generate result 
        model.obj = Objective(
        expr = obj_expr,
        sense = minimize
        )
        solver = SolverFactory('ipopt')
        results = solver.solve(model)
        # parse results and place into model 
        for i, bus in enumerate(self.test_case['bus']): 
            row[f'ip_vm_{i}'] = model.x[self.var_dict[f'bus_{i}'][0]+1].value
            row[f'ip_va_{i}'] = model.x[self.var_dict[f'bus_{i}'][1]+1].value
        for i, gen in enumerate(self.test_case['gen']): 
            row[f'ip_pg_{i}'] = model.x[self.var_dict[f'gen_{i}'][0]+1].value
            row[f'ip_qg_{i}'] = model.x[self.var_dict[f'gen_{i}'][1]+1].value
        return row 

    def create_data(self, row, ac_ip = True): 
        # each data point has 1) label, 2) value, 3) constraint value (only for constraints)
        data_point = [[i, 0, 0] for i in range(self.num_constrs + self.num_vars + 1)]
        base_mva = self.test_case['baseMVA']
        # warm start power flow
        for i, bus in enumerate(self.test_case['bus']): 
            vm = self.var_dict[f'bus_{i}'][0]
            data_point[vm][1] = 1
            pd, qd  = self.var_dict[f'busload_{i}']
            data_point[pd][1] = row[f'pd_{i}']/base_mva
            data_point[qd][1] = row[f'qd_{i}']/base_mva
        for i, gen in enumerate(self.test_case['gen']): 
            pg, qg = self.var_dict[f'gen_{i}']
            gen_stat = self.var_dict[f'genstat_{i}'][0]
            # warm-start pg and qg
            data_point[pg][1] = (self.test_case['gen'][i][idx_gen.PMAX]/2 * gen_stat)/base_mva
            data_point[qg][1] = (self.test_case['gen'][i][idx_gen.QMAX]/2 * gen_stat)/base_mva
            data_point[gen_stat][1] = row[f'status_{i}']
        # warm start line flows 
        for i, branch in enumerate(self.test_case['branch']): 
            pflow, qflow = self.warm_start_pf(i, data_point)
            pf, qf = self.var_dict[f'branch_{i}'][0:2]
            data_point[pf][1] = pflow
            data_point[qf][1] = qflow
    
        # warm start constraints 
        for constr_vals in self.constr_dict.values(): 
            for constr in constr_vals: 
                # obtain the constraint function 
                func = list(self.deriv_lookup[constr].values())[0]
                val = func(data_point[constr], data_point[0], data_point, self.lookup_helper) # constraint value 
                data_point[constr][1] = max(0, val)

        # prepare labels 
        if not ac_ip:
            self.num_predicted_values = self.num_vars - 2*len(self.test_case['branch']) - 2*len(self.test_case['bus']) - len(self.test_case['gen'])
            row_vals = [None] * self.num_predicted_values
            for i, bus in enumerate(self.test_case['bus']): 
                vm, va = self.var_dict[f'bus_{i}'][0:2]
                row_vals[vm] = [row[f'vm_{i}']]
                row_vals[va] =  [row[f'theta_{i}']]
            for i, gen in enumerate(self.test_case['gen']): 
                pg, qg = self.var_dict[f'gen_{i}']
                row_vals[pg] = [row[f'pg_{i}']/base_mva]
                row_vals[qg] = [row[f'qg_{i}']/base_mva]
        else: 
            self.num_predicted_values = self.num_vars - 2*len(self.test_case['branch']) - 2*len(self.test_case['bus']) - len(self.test_case['gen'])
            row_vals = [None] * self.num_predicted_values
            if 'ip_vm_0' not in row.columns:
                row = self.create_ipopt_model(row)
            for i, bus in enumerate(self.test_case['bus']): 
                vm, va = self.var_dict[f'bus_{i}'][0:2]
                row_vals[vm] = [row[f'ip_vm_{i}']]
                row_vals[va] =  [row[f'ip_va_{i}']]
            for i, gen in enumerate(self.test_case['gen']): 
                pg, qg = self.var_dict[f'gen_{i}']
                row_vals[pg] = [row[f'ip_pg_{i}']/base_mva]
                row_vals[qg] = [row[f'ip_qg_{i}']/base_mva]

        labels = torch.tensor(row_vals, dtype = torch.float)
        data = Data(x=torch.as_tensor(data_point, dtype=torch.float32), edge_index=self.opt_index, f_edge_index = self.feas_index, y = labels)
        return data
    
    def create_dataset(self, df): 
        self.dataset = [self.create_data(row, ac_ip = False) for _, row in df.iterrows()]
        
    def get_predictions(self, vars): 
        vals = [None] * self.num_predicted_values
        # obtain predictions 
        for i, bus in enumerate(self.test_case['bus']): 
            vm, va = self.var_dict[f'bus_{i}'][0:2]
            vals[vm] = [vars[vm][1]]
            vals[va] =  [vars[va][1]]
        for i, gen in enumerate(self.test_case['gen']): 
            pg, qg = self.var_dict[f'gen_{i}']
            vals[pg] = [vars[pg][1]]
            vals[qg] = [vars[qg][1]]
        return vals
    
    def get_objective(self, vars): 
        total_cost = 0
        for i, gencost in enumerate(self.test_case['gencost']): 
            pg = vars[self.var_dict[f'gen_{i}'][0]][1]
            cost_terms = gencost[4:][::-1] # cost terms from 0 power to highest power
            for c, cost in enumerate(cost_terms): 
                total_cost += cost * pg**c
        return total_cost 
    
    def get_violation(self, vars): 
        total_violation = 0 
        constr_inds = range(self.num_vars, self.num_vars + self.num_constrs + 1)
        # iterate through constraints 
        for constr in constr_inds: 
            constr_amt = vars[constr][2]
            total_violation += F.relu(constr_amt)
        return total_violation

    def warm_start_pf(self, line_ind, data_point):
        #  get branch information 
        branch = self.test_case['branch'][line_ind]
        G = branch[idx_brch.BR_R]/(branch[idx_brch.BR_R]**2 + branch[idx_brch.BR_X]**2)
        B = -1*branch[idx_brch.BR_X]/(branch[idx_brch.BR_R]**2 + branch[idx_brch.BR_X]**2)        # get relevant variable values 
        fbus, tbus = int(branch[idx_brch.F_BUS]), int(branch[idx_brch.T_BUS])
        vm_i, va_i = np.array(data_point)[self.var_dict[f'bus_{fbus}'][0:2], 1]
        vm_j, va_j = np.array(data_point)[self.var_dict[f'bus_{tbus}'][0:2],1]
        # compute pflow  and qflow
        pflow = vm_i **2 * G - vm_i * vm_j * (G*cos(va_i - va_j) + B * sin(va_i - va_j))
        qflow = -vm_i**2 * B - vm_i * vm_j * (G*sin(va_i - va_j) - B * cos(va_i - va_j))
        return pflow, qflow

    def translate_constr_dict(self, pickle = True):
        if pickle: 
            # replace functions with function names for pickling 
            for var, key_dict in self.deriv_lookup.items(): 
                for key, func in key_dict.items(): 
                    key_dict[key] = func.__name__
                self.deriv_lookup[var] = key_dict
            x = 10
        else: 
            # replace function names with function pointers 
            for var, key_dict in self.deriv_lookup.items(): 
                for key, func in key_dict.items(): 
                    key_dict[key] = name_to_func[func]
                self.deriv_lookup[var] = key_dict


def prepare_test_case(test_case): 
    tc_mapping = {bus[0]: j for (j, bus) in enumerate(test_case['bus'])}
    # re-index bus 
    test_case['bus'][:, idx_bus.BUS_I] = range(len(test_case['bus']))
    # re-index generators 
    test_case['gen'][:, idx_gen.GEN_BUS] = [tc_mapping[i] for i in test_case['gen'][:,idx_gen.GEN_BUS]]
    # re-index to and from on lines 
    test_case['branch'][:, idx_brch.T_BUS] = [tc_mapping[i] for i in test_case['branch'][:,idx_brch.T_BUS]]
    test_case['branch'][:, idx_brch.F_BUS] = [tc_mapping[i] for i in test_case['branch'][:,idx_brch.F_BUS]]
    return test_case

def create_and_store_model(CASE_NAME, test_case, data_df, model_pth): 
    # create model object
    pdgnn = PDGnn(test_case)
    # store dataframe in object 
    pdgnn.create_dataset(data_df)
    # prepare model for pickling 
    pdgnn.translate_constr_dict(pickle=True)
    # pickle the object for future runs 
    with open(os.path.join(model_pth, f'{CASE_NAME}_pdgnn.pkl'), "wb") as f:
        pickle.dump(pdgnn, f)

def test_model(pdgnn): 
    # create model 
    model = pdgnn.create_model('feasv3',  optim_lst = [[4, 0.1], [6, 0.05], [8, 0.025]],
                                feas_lst = [[3, 0.1], [5, 0.05], [7, 0.025]])
    # create dataset 
    data = pdgnn.dataset[0]
    out = model(data.x, data.edge_index, data.f_edge_index)
    x = 10

if __name__ == '__main__': 
    CASE_NAME = 'case6ww'
    tc_dict = {'case14': case14.case14(),
            'case6ww': case6ww.case6ww()}
    test_case = prepare_test_case(tc_dict[CASE_NAME])
    data_df = pd.read_excel(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'case6ww_acopf.xlsx'), index_col=None)

    model_pth = os.path.join(os.path.dirname(__file__), '../models/pdgnn_parents')
    create_and_store_model(CASE_NAME, test_case, data_df, model_pth)

    # # test making model 
    # pdgnn_test = PDGnn(test_case)
    # pdgnn_test.create_dataset(data_df)
    # test_model(pdgnn_test)
    x = 10