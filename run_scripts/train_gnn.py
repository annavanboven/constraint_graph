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
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data_df = pd.read_excel(os.path.join(os.path.dirname(__file__), "..", 'datasets', 'base_file.xlsx'))

def create_data(row):
    l1, l2 = _determine_init_violation_(row)
    # data point has 1) label, 2) value, 3) slope, 4) intercept (for this problem)
    point = torch.tensor([[1, 5, 0, 0],
            [2, 5, 0, 0],
            [3, l1, row['m1'], row['b1']],
            [4, l2, row['m2'], row['b2']]], dtype = torch.float)
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
    3: [0, 100],
    4: [0,100]
}
# create data point 
edge_index = EdgeIndex(
    [[0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3],
     [2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 2, 3]]
)
indices_dict = {
    0: [4, 6, 8],
    1: [5, 7, 9],
    2: [0, 2, 10],
    3: [1, 3, 11]
}
scale_val = 4

#  create graph object
gnn = PDConv(constr_dict, proj_dict, num_edges=12, 
             itrs=30, alpha = 0.1, convergence_tol= 0.001, indices_dict=indices_dict, scale=scale_val)

loss_obj = TestLoss()
def train(model, train_loader, optimizer, loss_obj):
    model.train()
    total_loss = 0
    n_dps = 0
    for (i, data) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_obj(out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_dps += 1
    return total_loss/n_dps


# separate data
data_list = [create_data(row) for _, row in data_df.iterrows()]
train_data, test_data = train_test_split(data_list, test_size=0.5, random_state=42)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)
# create optimizer
optimizer = torch.optim.Adam(gnn.parameters(), lr = 0.05)
# # train 
# loss_lst = []
# for epoch in range(1):
#     loss = train(gnn, train_loader, optimizer, loss_obj)
#     print(f" epoch: {epoch} \t loss: {loss} \n ")
#     loss_lst.append(loss)
#     if loss == 0:
#         break


def evaluate(model, loader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index)
            predictions.append([out[0][1], out[1][1]])
            labels.append(data.y.numpy())
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return mse, r2, predictions, labels

mse, r2, predictions, labels = evaluate(gnn, test_loader)
x = 10

