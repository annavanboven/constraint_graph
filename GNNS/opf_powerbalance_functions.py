
from torch import sin, cos, sqrt
from math import sin, cos
from pyomo.environ import cos, sin

def real_power_balance(vm, va, pg, pd, G, neighbor_vals): 
    val = G * vm**2 - (pg - pd)
    for neighbor in neighbor_vals: 
        vmn, van, G_ij, B_ij = neighbor
        val = val + vm * vmn * (G_ij * cos(va - van) + B_ij * sin(va - van))
    return val 

def reactive_power_balance(vm, va, qg, qd, B, neighbor_vals): 
    val = -B * vm**2 - (qg - qd)
    for neighbor in neighbor_vals:
        vmn, van, G_ij, B_ij = neighbor
        val += vm * vmn * (G_ij * sin(va - van) - B_ij * cos(va - van))
    return val


def dp_dp(p, l, vars, lookup_table): 
    return -1 * l[1] 

def dq_dq(q, l, vars, lookup_table): 
    return -1 * l[1] 

def dp_dv(v, l, vars, lookup_table):
    if 0 in [v[0], l[0]]:
        x = 10
    bus_lst = lookup_table[v[0]][l[0]] # for this constraint, bus_lst is [vm_ind, va_ind, real, imaginary]
    vthet = bus_lst[0] # first value on list is [ind_of_vm, ind_of_bustheta, real admittance]
    va = vars[vthet[1]][1]
    val = 2 * vthet[2] * v[1]
    # sum over neighbors
    for bus in bus_lst[1:]: 
        vm_ind, va_ind, real_ad, im_ad = vars[bus[0]][1], vars[bus[1]][1], bus[2], bus[3]
        val = val + vm_ind * (real_ad * cos(va - va_ind) + im_ad * sin(va - va_ind))
    return l[1] * val 

def dp_dtheta(v, l, vars, lookup_table): 
    bus_lst = lookup_table[v[0]][l[0]] # for this constraint, bus_lst is [vm_ind, va_ind, real, imaginary]
    vthet = bus_lst[0] # first value on list is [ind_of_vm, ind_of_bustheta, real admittance]
    va = v[1]
    val = vars[vthet[0]][1]
    for bus in bus_lst[1:]: 
        vm_ind, va_ind, real_ad, im_ad = vars[bus[0]][1], vars[bus[1]][1], bus[2], bus[3]
        val = val + vm_ind * (real_ad *-1 * sin(va - va_ind) + im_ad * cos(va - va_ind))
    return l[1] * val

def dq_dv(v, l, vars, lookup_table): 
    bus_lst = lookup_table[v[0]][l[0]] # for this constraint, bus_lst is [vm_ind, va_ind, real, imaginary]
    vthet = bus_lst[0] # first value on list is [ind_of_vm, ind_of_bustheta, imag admittance]
    va = vars[vthet[1]][1]
    val = 2 * vthet[2] * v[1]
    # sum over neighbors
    for bus in bus_lst[1:]: 
        vm_ind, va_ind, real_ad, im_ad = vars[bus[0]][1], vars[bus[1]][1], bus[2], bus[3]
        val = val + vm_ind * (-1 * im_ad * cos(va - va_ind) + real_ad * sin(va - va_ind))
    return l[1] * val 

def dq_dtheta(v, l, vars, lookup_table): 
    bus_lst = lookup_table[v[0]][l[0]] # for this constraint, bus_lst is [vm_ind, va_ind, real, imaginary]
    vthet = bus_lst[0] # first value on list is [ind_of_vm,ind_of_bustheta]
    va = v[1]
    val = vars[vthet[0]][1]
    for bus in bus_lst[1:]: 
        vm_ind, va_ind, real_ad, im_ad = vars[bus[0]][1], vars[bus[1]][1], bus[2], bus[3]
        val = val + vm_ind * (im_ad *-1 * sin(va - va_ind) + real_ad * cos(va - va_ind))
    return l[1] * val

def dpn_dv(v, l, vars, lookup_table): 
    lookup = lookup_table[v[0]][l[0]] # for this constraint, bus_lst is [vm_ind, va_ind, real, imaginary]
    va = vars[lookup[0]][1] # first value is [ind_of_bustheta]
    neighbor = lookup
    va_n, real_ad, im_ad = vars[neighbor[1]][1], neighbor[2], neighbor[3]
    val = v[1] * (real_ad * cos(va - va_n) + im_ad * sin(va - va_n))
    return l[1] * val

def dpn_dtheta(v, l, vars, lookup_table): 
    lookup = lookup_table[v[0]][l[0]] # for this constraint, bus_lst is [vm_ind,  va_ind, real, imaginary]
    va = vars[lookup[0]][1] # first value is [ind_of_bustheta]
    neighbor = lookup
    vm_n, va_n, real_ad, im_ad = vars[neighbor[0]][1], vars[neighbor[1]][1], neighbor[2], neighbor[3]
    val = v[1] *  vm_n * (real_ad * sin(va - va_n) + im_ad * -cos(va - va_n))
    return l[1] * val

def dqn_dv(v, l, vars, lookup_table): 
    lookup = lookup_table[v[0]][l[0]] # for this constraint, bus_lst is [vm_ind, va_ind, real, imaginary]
    va = vars[lookup[0]][1] # first value is [ind_of_bustheta]
    neighbor = lookup
    va_n, real_ad, im_ad = vars[neighbor[1]][1], neighbor[2], neighbor[3]
    val = v[1] * (-1 * im_ad * cos(va - va_n) + real_ad * sin(va - va_n))
    return l[1] * val

def dqn_dtheta(v, l, vars, lookup_table): 
    lookup = lookup_table[v[0]][l[0]] # for this constraint, bus_lst is [vm_ind,  va_ind, real, imaginary]
    va = vars[lookup[0]][1] # first value is [ind_of_bustheta]
    neighbor = lookup
    vm_n, va_n, real_ad, im_ad = vars[neighbor[0]][1], vars[neighbor[1]][1], neighbor[2], neighbor[3]
    val = v[1] *  vm_n * (-im_ad * sin(va - va_n) + real_ad * -cos(va - va_n))
    return l[1] * val

def pbal_lambda(l, v, vars, lookup_table):
    if 0 in [l[0], v[0]]:
        x = 10
    neighbor_helpers = lookup_table[l[0]] 
    vm_ind, va_ind, real_ad, pg_inds, pd_ind, _, _ = neighbor_helpers[0] # indices for the bus itself 
    vm, va, pg, pd = vars[vm_ind][1], vars[va_ind][1], sum([vars[pg_ind][1] for pg_ind in pg_inds]), vars[pd_ind][1]
    val = real_ad * vm**2 - (pg - pd)
    for neighbor in neighbor_helpers[1:]: 
        vmn, van, real_ad, im_ad = vars[neighbor[0]][1], vars[neighbor[1]][1], neighbor[2], neighbor[3]
        val = val + vm * vmn * (real_ad * cos(va - van) + im_ad * sin(va - van))
    # store current equation value 
    vars[int(l[0])][2] = val
    return val

def qbal_lambda(l, v, vars, lookup_table): 
    neighbor_helpers = lookup_table[l[0]] 
    vm_ind, va_ind, real_ad,_, _, qg_inds, qd_ind = neighbor_helpers[0] # indices for the bus itself 
    vm, va, qg, qd = vars[vm_ind][1], vars[va_ind][1], sum([vars[qg_ind][1] for qg_ind in qg_inds]), vars[qd_ind][1]
    val = real_ad * vm**2 - (qg - qd)
    for neighbor in neighbor_helpers[1:]: 
        vmn, van, real_ad, im_ad = vars[neighbor[0]][1], vars[neighbor[1]][1], neighbor[2], neighbor[3]
        val = val + vm * vmn * (real_ad * sin(va - van) - im_ad * cos(va - van))
    # store current equation value 
    vars[int(l[0])][2]  = val
    return val

def empty_lambda(v, l, vars, lookup_table): 
    return 0