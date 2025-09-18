import sys, os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GNNS.GNN_classifier import PDConv
from GNNS.loss_functions import TestLoss
from GNNS.opt_functions import *
import torch
from math import sin, cos
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn.conv import MessagePassing
import os
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition

def generate_problem(model, 
                     slope_1, vert_1, slope_2, vert_2):
    constraints = [
        sin(model.x) + slope_1*model.x  + vert_1 <= model.y,
        cos(model.x) + slope_2*model.x  - vert_2 >= model.y
    ]
    # add constraints to model
    model.constraints = ConstraintList()
    for con in constraints:
        model.constraints.add(con)
model = ConcreteModel()
model.x = Var(bounds=(0, 10), domain = NonNegativeReals)
model.y = Var(bounds=(0, 10), domain = NonNegativeReals)
generate_problem(model, 3, -2, 4, -3)
model.obj = Objective(
    expr = model.y + 3*model.x,
    sense = minimize
    )
solver = SolverFactory('ipopt')
start_time = time.time()
results = solver.solve(model,)
end_time = time.time()
true_time = end_time - start_time