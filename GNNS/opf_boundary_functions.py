def upper_lower_vm_boundaries(vm, vmin, vmax): 
    return [vmin - vm, vm - vmax]

def upper_lower_gen_boundaries(gen, genmin, genmax, genstat): 
    return [genmin - genstat * gen <= 0, genstat * gen - genmax <= 0]

# boundary functions 
def boundary_min(x, l, vars, lookup_table): 
    return l[1] 

def boundary_max(x, l, vars, lookup_table): 
    return -1 * l[1] 

def boundary_lambda_vmin(l, x, vars, lookup_table): 
    vmin  = lookup_table[l[0]] 
    vars[int(l[0])][2] = vmin  - x[1]
    return vmin  - x[1]

def boundary_lambda_vmax(l, x, vars, lookup_table): 
    vmax  = lookup_table[l[0]] 
    vars[int(l[0])][2] = x[1] - vmax
    return x[1] - vmax

def boundary_lambda_pmin(l, x, vars, lookup_table): 
    pmin, gen_stat = lookup_table[l[0]] # for this constraint, store [min, generator status]
    gen_stat = vars[gen_stat][1]
    vars[int(l[0])][2] = pmin * gen_stat - x[1]
    return pmin * gen_stat - x[1]

def boundary_lambda_pmax(l, x, vars, lookup_table): 
    pmax, gen_stat = lookup_table[l[0]]
    gen_stat = vars[gen_stat][1]
    vars[int(l[0])][2] = x[1] - gen_stat * pmax
    return x[1] - gen_stat * pmax
