import torch
from torch import sin, cos
import numpy as np
from torch_geometric.nn.conv import MessagePassing

import torch.nn as nn
import torch.nn.functional as F

def x1x1(x_i, x_j):
    return -1

def x2x2(x_i, x_j): 
    return -3

def c1x1(x_1, lambda_1):
    return -1 * lambda_1[1] * (cos(x_1[1]) + lambda_1[2])

def c2x1(x_1, lambda_2):
    return -1 * lambda_2[1] * (sin(x_1[1]) - lambda_2[2])

def c1x2(x_2, lambda_1):
    return lambda_1[1]

def c2x2(x_2, lambda_2):
    return -1 * lambda_2[1]

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

def feas(x):
    x1, x2, l1, l2 = x[0], x[1], x[2], x[3]
    return (sin(x1[1]) + l1[2]*x1[1] + l1[3] - x2[1] <= 0) and (-1*cos(x1[1]) - l2[2]*x1[1] - l2[3] + x2[1] <= 0)