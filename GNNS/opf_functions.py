import torch
from torch import sin, cos
import numpy as np
from torch_geometric.nn.conv import MessagePassing

import torch.nn as nn
import torch.nn.functional as F
lookup_table = {}
# objective function values 
def variable_objective(x1, x2, vars): 
    return lookup_table[x1[0]]

# boundary functions 
def boundary_min(x, l, vars): 
    return l 

def boundary_max(x, l, vars): 
    return -1 * l 

def boundary_lambda_min(x, l, vars): 
    return x[1] - lookup_table[l[0]]

def boundary_lambda_max(x, l, vars): 
    return lookup_table[l[0]] - x[1]

# line flow functions CHECK THESE AND ADD LIMITS
def dflow_dv(x, l, vars): 
    y_ij, v_ind = lookup_table[l[1]]
    v_j = vars[v_ind][1]
    return l[1] * y_ij * 2 * x[1] - y_ij * v_j

def flow(x, l, vars): 
    y_ij, v_ind = lookup_table[l[1]]
    v_j = vars[v_ind][1]
    return y_ij * x[1]**2 - 0.5 * y_ij * x[1] * v_j 




# power balance functions 
def dp_dp(p, l, vars): 
    return -1 * l 

def dq_dq(q, l, vars): 
    return -1 * l 

def dp_dv(v, l, vars):
    bus_lst = lookup_table[v[0]][l[0]]
    val = 0
    for bus in bus_lst: 
        val += vars[bus[0]][1] * (bus[1] * cos(vars[bus[2]][1] - vars[bus[3]][1]) + bus[4] * sin(vars[bus[2]][1] - vars[bus[3]][1]))
    return l[1] * val 