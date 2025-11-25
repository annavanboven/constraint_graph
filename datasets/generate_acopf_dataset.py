# %% imports
import os 
import pandas as pd
import numpy as np 
from pypower import  case14, runopf 
import random 
from pypower.idx_bus import *
from pypower.idx_brch import *
from pypower.idx_gen import *

test_case = case14.case14()
CASE_NAME = 'case14'
# %% functions 
def perturb_gen(test_case): 
    # find number of generators to be online 
    total_gen = len(test_case['gen'])
    num_gen = int(random.uniform(0.5, 1) * total_gen)
    on_gens = random.sample(range(total_gen), num_gen)
    # place generators online and return total maxgen 
    maxgen = 0
    for (i, gen) in enumerate(test_case['gen']): 
        gen[GEN_STATUS] = i in on_gens 
        maxgen += gen[PMAX] if i in on_gens else 0
    return test_case, maxgen 

def perturb_load(test_case, maxgen, total_gen): 
    # perturb load 
    load_buses = np.array([i for i in range(len(test_case['bus'])) if test_case['bus'][i][PD] > 0])
    n_loads = len(load_buses)
    delta = 1 - maxgen/total_gen 
    perturbation = [random.uniform(1 - delta, 1) for _ in range(n_loads)]
    for perturb, bus in zip(perturbation, load_buses): 
        test_case['bus'][bus][PD] = test_case['bus'][bus][PD] * perturb 
    while sum(test_case['bus'][load_buses, PD]) > maxgen + 0.05: 
        perturbation = [random.uniform(1 - delta, 1 + delta) for _ in range(n_loads)]
        for perturb, bus in zip(perturbation, load_buses): 
            test_case['bus'][bus][PD] = test_case['bus'][bus][PD] * perturb 
    return test_case

def parse_sample(sample_dict, result, test_case): 
    # store cost 
    sample_dict['cost'].append(result['f'])
    # store bus values 
    for i, bus in enumerate(test_case['bus']): 
        sample_dict[f'pd_{i}'].append(bus[PD])
        sample_dict[f'qd_{i}'].append(bus[QD])
        sample_dict[f'vm_{i}'].append(result['var']['val']['Vm'][i])
        sample_dict[f'theta_{i}'].append(result['var']['val']['Va'][i])
    # store generator values 
    for i, gen in enumerate(test_case['gen']): 
        sample_dict[f'status_{i}'].append(gen[GEN_STATUS])
        sample_dict[f'pg_{i}'].append(result['gen'][i, PG])
        sample_dict[f'qg_{i}'].append(result['gen'][i, QG])
    return sample_dict

# %% generate random sample 
test_case, maxgen = perturb_gen(test_case)
test_case = perturb_load(test_case, maxgen, sum(test_case['gen'][:, PMAX]))
# %% generate dataset
num_points = 10
base_load = test_case['bus'][:, PD].copy()
total_gen = sum(test_case['gen'][:, PMAX])
sample_dict = {'cost': []}
for i, bus in enumerate(test_case['bus']): 
    sample_dict[f'pd_{i}'] = []
    sample_dict[f'qd_{i}'] = []
    sample_dict[f'vm_{i}'] = []
    sample_dict[f'theta_{i}'] = []
for i, gen in enumerate(test_case['gen']): 
    sample_dict[f'pg_{i}'] = []
    sample_dict[f'status_{i}'] = []
    sample_dict[f'qg_{i}'] = []
for _ in range(num_points): 
    # reset test case 
    test_case['bus'][:, PD] = base_load
    test_case['gen'][:, GEN_STATUS] = [1] * len(test_case['gen'])
    # perturb generation and load 
    test_case, maxgen = perturb_gen(test_case)
    test_case = perturb_load(test_case, maxgen, total_gen)
    # run acopf 
    result = runopf.runopf(test_case)
    # store sample 
    sample_dict = parse_sample(sample_dict, result, test_case)
# %% output dataset 
sample_df = pd.DataFrame(sample_dict)
sample_df.to_excel(f'{CASE_NAME}_acopf.xlsx')

# %%
