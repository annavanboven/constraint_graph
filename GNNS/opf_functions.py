import torch
from torch import sin, cos, sqrt
import numpy as np
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from GNNS.opf_boundary_functions import * 
from GNNS.opf_linelimit_functions import * 
from GNNS.opf_powerbalance_functions import *
# objective function values 
def variable_objective(x1, x2, vars, lookup_table): 
    cost_terms = lookup_table[x1[0]][x1[0]]
    gen = x1[1]
    return min_cost(gen, cost_terms)

def min_cost(gen, cost_terms):
    val = 0 
    for i, cost in enumerate(cost_terms): 
        val += cost * gen**i
    return val


def constraint_feasibility(vars, first_constr):
    constrs = vars[first_constr:-1, :]
    infeas = sum([max(0, c[2]) for c in constrs])
    return infeas.item() 