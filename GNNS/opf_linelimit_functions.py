from torch import sin, cos
from math import sin, cos, sqrt
from pyomo.environ import cos, sin


def qflow_constraint(vm_i, va_i, vm_j, va_j, G, B):
    return -vm_i**2 * B - vm_i * vm_j * (G*sin(va_i - va_j) - B * cos(va_i - va_j))

def pflow_constraint(vm_i, va_i, vm_j, va_j, G, B): 
    return vm_i **2 * G - vm_i * vm_j * (G*cos(va_i - va_j) + B * sin(va_i - va_j))

def apparent_flow_constraint(pflow, qflow, thermal_limit):
    return pflow**2 + qflow**2 - thermal_limit**2

# thermal limits 
def pflow_lambda(l, v, vars, lookup_table): 
    bus1, bus2, p_ind, line_chars = lookup_table[l[0]]
    vm_i, va_i = vars[bus1[0]][1], vars[bus1[1]][1] # bus i values
    vm_j, va_j = vars[bus2[0]][1], vars[bus2[1]][1] # bus j values
    pflow = vars[p_ind][1] # real flow value
    G, B = line_chars # line characteristics 
    val = pflow_constraint(vm_i, va_i, vm_j, va_j, G, B)
    # update lambda value 
    vars[int(l[0])][2]  = val - pflow
    return val - pflow

def qflow_lambda(l, v, vars, lookup_table): 
    bus1, bus2, q_ind, line_chars = lookup_table[l[0]]
    vm_i, va_i = vars[bus1[0]][1], vars[bus1[1]][1] # bus i values
    vm_j, va_j = vars[bus2[0]][1], vars[bus2[1]][1] # bus j values
    qflow = vars[q_ind][1]
    G, B = line_chars # line characteristics 
    val = qflow_constraint(vm_i, va_i, vm_j, va_j, G, B)
    # update lambda value 
    vars[int(l[0])][2]  = val - qflow
    return val - qflow

def dpflow_dvi(v, l, vars, lookup_table): 
    bus1, bus2, p_ind, line_chars = lookup_table[v[0]][l[0]]
    vm_i, va_i = v[1], vars[bus1[1]][1] # bus i values
    vm_j, va_j = vars[bus2[0]][1], vars[bus2[1]][1] # bus j values
    pflow = vars[p_ind][1] # real flow value
    G, B = line_chars # line characteristics 
    val = vm_i*2 * G - vm_j * (G*cos(va_i - va_j) + B * sin(va_i - va_j))
    return val * l[1]

def dqflow_dvi(v, l, vars, lookup_table): 
    bus1, bus2, q_ind, line_chars = lookup_table[v[0]][l[0]]
    vm_i, va_i = v[1], vars[bus1[1]][1] # bus i values
    vm_j, va_j = vars[bus2[0]][1], vars[bus2[1]][1] # bus j values
    qflow = vars[q_ind][1] # real flow value
    G, B = line_chars # line characteristics 
    val = -2*vm_i * B - vm_j * (G*sin(va_i - va_j) - B * cos(va_i - va_j))
    return val * l[1]

def dpflow_dvj(v, l, vars, lookup_table): 
    bus1, bus2, p_ind, line_chars = lookup_table[v[0]][l[0]]
    vm_i, va_i = vars[bus1[0]][1], vars[bus1[1]][1] # bus i values
    vm_j, va_j = v[1], vars[bus2[1]][1] # bus j values
    pflow = vars[p_ind][1] # real flow value
    G, B = line_chars # line characteristics 
    val =  -vm_i * (G*cos(va_i - va_j) + B * sin(va_i - va_j))
    return val * l[1]

def dqflow_dvj(v, l, vars, lookup_table): 
    bus1, bus2, q_ind, line_chars = lookup_table[v[0]][l[0]]
    vm_i, va_i = vars[bus1[0]][1], vars[bus1[1]][1] # bus i values
    vm_j, va_j = v[1], vars[bus2[1]][1] # bus j values
    qflow = vars[q_ind][1] # real flow value
    G, B = line_chars # line characteristics 
    val = -vm_i * (G*sin(va_i - va_j) - B * cos(va_i - va_j))
    return val * l[1]

def dpflow_dthetai(v, l, vars, lookup_table): 
    bus1, bus2, p_ind, line_chars = lookup_table[v[0]][l[0]]
    vm_i, va_i = vars[bus1[0]][1], vars[bus1[1]][1] # bus i values
    vm_j, va_j = vars[bus2[0]][1], vars[bus2[1]][1] # bus j values
    pflow = vars[p_ind][1] # real flow value
    G, B = line_chars # line characteristics 
    val = -vm_i * vm_j * (-G * sin(va_i - va_j) + B * cos(va_i - va_j))
    return val * l[1]

def dpflow_dthetaj(v, l, vars, lookup_table): 
    bus1, bus2, p_ind, line_chars = lookup_table[v[0]][l[0]]
    vm_i, va_i = vars[bus1[0]][1], vars[bus1[1]][1] # bus i values
    vm_j, va_j = vars[bus2[0]][1], vars[bus2[1]][1] # bus j values
    pflow = vars[p_ind][1] # real flow value
    G, B = line_chars # line characteristics 
    val = -vm_i * vm_j * (G * sin(va_i - va_j) - B * cos(va_i - va_j))
    return val * l[1]

def dqflow_dthetai(v, l, vars, lookup_table): 
    bus1, bus2, p_ind, line_chars = lookup_table[v[0]][l[0]]
    vm_i, va_i = vars[bus1[0]][1], vars[bus1[1]][1] # bus i values
    vm_j, va_j = vars[bus2[0]][1], vars[bus2[1]][1] # bus j values
    pflow = vars[p_ind][1] # real flow value
    G, B = line_chars # line characteristics 
    val = -vm_i * vm_j * (G * cos(va_i - va_j) + B * sin(va_i - va_j))
    return val * l[1]

def dqflow_dthetaj(v, l, vars, lookup_table): 
    bus1, bus2, p_ind, line_chars = lookup_table[v[0]][l[0]]
    vm_i, va_i = vars[bus1[0]][1], vars[bus1[1]][1] # bus i values
    vm_j, va_j = vars[bus2[0]][1], vars[bus2[1]][1] # bus j values
    pflow = vars[p_ind][1] # real flow value
    G, B = line_chars # line characteristics 
    val = -vm_i * vm_j * (-G * cos(va_i - va_j) - B * sin(va_i - va_j))
    return val * l[1]

def dpflow_dp(v, l, vars, lookup_table): 
    return l[1] * -1 
 
def dqflow_dq(v, l, vars, lookup_table): 
    return l[1] * -1

def apparent_flow_lambda(l, v, vars, lookup_table): 
    pflow_ind, qflow_ind, thermal_limit = lookup_table[l[0]]
    pflow, qflow = vars[pflow_ind][1], vars[qflow_ind][1]
    val = apparent_flow_constraint(pflow, qflow, thermal_limit)
    vars[int(l[0])][2]  = val 
    return val 

def dapparent_flow(v, l, vars, lookup_table): 
    return 2 * v[1] * l[1]
