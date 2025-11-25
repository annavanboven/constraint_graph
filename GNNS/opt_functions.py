import torch
from torch import sin, cos
import numpy as np
from torch_geometric.nn.conv import MessagePassing

import torch.nn as nn
import torch.nn.functional as F

def x1x1(x_i, x_j):
    # return 0
    return -1

def x2x2(x_i, x_j): 
    # return 0
    return -3

def c1x1(x_1, lambda_1):
    # return -1 * lambda_1[1] * (cos(x_1[1]) + lambda_1[2])
    deriv = (cos(x_1[1]) + lambda_1[2])
    return -1 * deriv * (lambda_1[1] + lambda_1[4])

def c2x1(x_1, lambda_2):
    # return -1 * lambda_2[1] * (sin(x_1[1]) - lambda_2[2])
    deriv = (sin(x_1[1]) - lambda_2[2])
    return -1 * deriv * (lambda_2[1] + lambda_2[4])

def c1x2(x_2, lambda_1):
    # return lambda_1[1]
    deriv = -1
    return -1 * deriv * (lambda_1[1] + lambda_1[4])
    

def c2x2(x_2, lambda_2):
    # return -1 * lambda_2[1]
    deriv = 1
    return -1 * deriv * (lambda_2[1] + lambda_2[4])

def x1l1(l1, x_1):
    return sin(x_1[1]) + l1[2]*x_1[1]

def x2l1(l1, x_2): 
    return -1 * x_2[1]

def x1l2(l2, x_1):
    return -1*cos(x_1[1]) - l2[2]*x_1[1]

def x2l2(l2, x_2):
    return x_2[1]

def l1l1(l1, l1j):
    return l1[3]

def l2l2(l2, l2j):
    return -1*l2[3]

def feas(x, x1_inds, x2_inds, l1_inds, l2_inds, epsilon = 0.001):
    x1 = x[x1_inds, :]
    x2 = x[x2_inds, :]
    l1 = x[l1_inds, :]
    l2 = x[l2_inds, :]
    c1 = sin(x1[:, 1]) + l1[:, 2]*x1[:, 1] + l1[:, 3] - x2[:, 1] <= 0 + epsilon
    c2 = -1*cos(x1[:, 1]) - l2[:, 2]*x1[:, 1] - l2[:, 3] + x2[:, 1] <= 0 + epsilon
    is_feas = c1 & c2
    num_true = torch.sum(is_feas).item()
    percent_true = num_true/len(is_feas)
    return percent_true

def amt_feas(x, x1_inds, x2_inds, l1_inds, l2_inds):
    x1 = x[x1_inds, :]
    x2 = x[x2_inds, :]
    l1 = x[l1_inds, :]
    l2 = x[l2_inds, :]
    c1 = sin(x1[:, 1]) + l1[:, 2]*x1[:, 1] + l1[:, 3] - x2[:, 1]
    c2 = -1*cos(x1[:, 1]) - l2[:, 2]*x1[:, 1] - l2[:, 3] + x2[:, 1]
    t = F.leaky_relu(c1)
    m = F.leaky_relu(c2)
    amt_feas = torch.sum(F.leaky_relu(c1)) + torch.sum(F.leaky_relu(c2))
    return amt_feas