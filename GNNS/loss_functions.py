import torch
from torch import sin, cos
import torch.nn as nn
import torch.nn.functional as F

class TestLoss(nn.Module):
    def __init__(self, obj_scale, constr_scale):
        super(TestLoss, self).__init__()
        self.s_obj = obj_scale
        self.s_constr = constr_scale


    def forward(self, inputs):
        # x1, x2, l1, l2 = inputs[:, 1]
        # m1, m2 = inputs[[2, 3], 2]
        # b1, b2 = inputs[[2, 3], 3]
        x1, x2, l1, l2 = inputs[0, 1], inputs[1, 1], inputs[2, 1], inputs[3, 1]
        m1, m2 = inputs[2, 2], inputs[3, 2]
        b1, b2 = inputs[2, 3], inputs[3, 3]
        # objective loss
        obj_loss = x1 + 3*x2
        # constraint loss 
        c1_loss = l1*F.relu((sin(x1) + m1*x1 + b1 - x2))
        c2_loss = l2*F.relu((-cos(x1) - m2*x1 + b2 + x2))
        # c1_loss = (sin(x1) + m1*x1 + b1 - x2)
        # c2_loss = (-cos(x1) - m2*x1 + b2 + x2)
        # return sum 
        return self.s_obj*obj_loss + self.s_constr*(l1*c1_loss + l2*c2_loss)
        # return obj_loss + c1_loss + c2_loss
