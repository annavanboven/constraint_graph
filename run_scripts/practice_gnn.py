#  imports
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
df = pd.read_excel(os.path.join(os.path.dirname(__file__), "..", 'datasets', 'base_file.xlsx'))

def create_data(row):
    l1, l2 = _determine_init_violation_(row)
    # data point has 1) label, 2) value, 3) slope, 4) intercept (for this problem), 5) constraint value
    point = torch.tensor([[1, 5, 0, 0, 0],
            [2, 5, 0, 0, 0],
            [3, l1, row['m1'], row['b1'], l1],
            [4, l2, row['m2'], row['b2'], l2]], dtype = torch.float)
    label = torch.tensor([[row['xopt']], [row['yopt']]], dtype=torch.float)
    data = Data(x=point, edge_index=edge_index, y = label)
    return data

def _determine_init_violation_(row): 
    # how much is the first constraint violated?
    l1 = max(sin(5) + row['m1']*5 + row['b1'] - 5, 0)
    # how much is the second constraint violated?
    l2 = max(-cos(5) - row['m2']*5 + row['b2'] + 5, 0)
    return l1, l2
#  setup
constr_dict = {
    '1-3': c1x1,
    '1-4': c2x1,
    '2-3': c1x2,
    '2-4': c2x2,
    '3-1': x1l1,
    '3-2': x2l1,
    '4-1': x1l2,
    '4-2': x2l2,
    '1-1': x1x1,
    '2-2': x2x2,
    '3-3': l1l1,
    '4-4': l2l2
    }

proj_dict = {
    1: [0, 10],
    2: [0,10],
    3: [0, 10],
    4: [0,10]
}
indices_dict = {
    0: [4, 6, 8],
    1: [5, 7, 9],
    2: [0, 2, 10],
    3: [1, 3, 11]
}
# create data point 
edge_index = EdgeIndex(
    [[0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3],
     [2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 2, 3]]
)
# test datapoint 
test_point = torch.tensor([[1, 4, 0, 0, 0],
            [2, 4, 0, 0, 0],
            [3, 0, 1, -1, 0],
            [4, 0, 2, -2, 0]], dtype = torch.float)
test_data = Data(x = test_point, edge_index=edge_index)

#  create graph object
alpha = .1
n_itrs = 200
scale_val = 1.15
conv_tol = 0.001
gnn = PDConv(constr_dict, proj_dict, indices_dict=indices_dict, num_edges=12, 
             itrs=n_itrs, alpha = alpha, convergence_tol= conv_tol, scale = scale_val)

# # test forward pass 
# start_time = time.time()
# output = gnn(test_data.x, test_data.edge_index)
# end_time = time.time()
# gnn_time = end_time - start_time



# test functions
# feas_point: x1, x2, l1, l2 = 1, 4, 0, 0
# infeas point: x1, x2, l1, l2 = 5, 5, 7.04, 0
m1 = 1
b1 = -1
m2 = 2
b2 = -2
a = 0.9
b = 0.1
n_itrs = 200
alpha = 0.05
def update_funcs(x1, x2, l1, l2,  s1, s2, a, b):
    # x1n = x1 - alpha*(1 + (l1 + c1)*(cos(x1) + m1) + (l2 + c2)*(sin(x1) - m2))
    # x2n = x2 - alpha*(3 - (l1 + c1) + (l2 + c2))
    x1n = x1 - alpha * (a + l1*(cos(x1) + m1) + l1*(sin(x1) - m2))
    x2n = x2 - alpha * (3*a - l1 + l2)
    s1n = s1 - alpha * (-1*b + l1 )
    s2n = s2 - alpha * (-1 * b + l2 )
    l1 += alpha*(sin(x1) + m1*x1 + b1 - x2 + s1)
    l2 += alpha*(-cos(x1) - m2*x1 - b2 + x2 + s2)
    a = max(a - 0.0025, 0.1)
    b = min(b + 0.0025, 0.9)
    return max(x1n, 0), max(x2n, 0), max(l1, 0), max(l2, 0),  max(s1n, 0), max(s2n, 0), a, b

x1, x2, l1, l2,  s1, s2 = 4, 4, 2, 2,  1.76, 1.35
storage = [[x1, x2, l1, l2,  s1, s2]]
start_time = time.time()
for _ in range(n_itrs):
    x1, x2, l1, l2,  s1, s2, a, b = update_funcs(x1, x2, l1, l2,  s1, s2, a, b)
    storage.append([x1, x2, l1, l2,  s1, s2])
    # if abs(storage[-1][0] - storage[-2][0]) + abs(storage[-1][1] - storage[-2][1]) < conv_tol:
    #     break
end_time = time.time()
baseline_time = end_time - start_time

#  plot 
lst = storage
import matplotlib.pyplot as plt
x1s = [i[0] for i in lst]
x2s = [i[1] for i in lst]
l1s = [i[2] for i in lst]
l2s = [i[3] for i in lst]
s1s = [i[4] for i in lst]
s2s = [i[5] for i in lst]
c_list = []

for (x1, x2) in zip(x1s, x2s):
    if sin(x1) + m1*x1 + b1 - x2 > 0:
        color = 'red'
        if -1*cos(x1)  -m2*x1 - b2 + x2 > 0:
            color = 'purple'
    elif -1*cos(x1) -m2*x1 - b2 + x2 > 0:
        color = 'blue'
    else:
        color = 'limegreen'
    c_list.append(color)
plt.scatter(range(len(x1s)), x1s, color = c_list, label = 'x1', marker = 'x',  alpha = 0.3)
plt.scatter(range(len(x2s)), x2s, color = c_list, label = 'x2', marker = 'D',  alpha = 0.3)
plt.plot(range(len(s1s)), s1s, color = 'red', label = 's1', linestyle = ':',  alpha = 0.3)
plt.plot(range(len(s2s)), s2s, color = 'blue', label = 's2', linestyle = '--',  alpha = 0.3)
plt.plot(range(len(l1s)), l1s, color = 'red', label = 'l1',  alpha = 0.3)
plt.plot(range(len(l2s)), l2s, color = 'blue', label = 'l2', alpha = 0.3)
plt.legend()
plt.show()


# for (i, row) in df.iterrows():
#     if i >= 6:
#         break
#     test_data = create_data(row)
#     output = gnn(test_data.x, test_data.edge_index)
#     lst = gnn.prev_vals
#     x1s = [i[0] for i in lst]
#     x2s = [i[1] for i in lst]
#     l1s = [i[2] for i in lst]
#     l2s = [i[3] for i in lst]
#     c_list = []
#     m1 = row['m1']
#     b1 = row['b1']
#     m2 = row['m2']
#     b2 = row['b2']
#     for (x1, x2) in zip(x1s, x2s):
#         if sin(x1) + m1*x1 + b1 - x2 > 0:
#             color = 'red'
#             if -1*cos(x1)  -m2*x1 - b2 + x2 > 0:
#                 color = 'purple'
#         elif -1*cos(x1) -m2*x1 - b2 + x2 > 0:
#             color = 'blue'
#         else:
#             color = 'limegreen'
#         c_list.append(color)
#     x1opt = [row['xopt'] for _ in range(len(x1s))]
#     x2opt = [row['yopt'] for _ in range(len(x2s))]
#     plt.scatter(range(len(x1s)), x1s, color = c_list, label = 'x1', marker = 'x')
#     plt.scatter(range(len(x2s)), x2s, color = c_list, label = 'x2', marker = 'D')
#     plt.plot(range(len(x1s)), x1opt, label = 'x1opt', linestyle = '--')
#     plt.plot(range(len(x2s)), x2opt, label = 'x2opt', linestyle = ':')
#     plt.plot(range(len(l1s)), l1s, color = 'red', label = 'l1',  alpha = 0.3)
#     plt.plot(range(len(l2s)), l2s, color = 'blue', label = 'l2', alpha = 0.3)
#     plt.legend()
#     plt.show()
#     x = 10

#  compare prev_vals with storage
min_size = min(len(storage), len(gnn.prev_vals))
st = np.array(storage)[0:min_size]
pv = np.array(gnn.prev_vals)[0:min_size]
diff = (pv - st).round(3)
x = 10
