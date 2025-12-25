# %% imports
import gurobipy as gp
import pandas as pd
import numpy as np
import os
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
#  problem generation
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



# %% test model creation
model = ConcreteModel()
model.x = Var(bounds=(0, 10), initialize = 5, domain = NonNegativeReals)
model.y = Var(bounds=(0, 10),initialize = 5,  domain = NonNegativeReals)
generate_problem(model, 3, -2, 4, -3)
# add objective
model.obj = Objective(
expr = model.y + 3*model.x,
sense = minimize
)
solver = SolverFactory('ipopt')
results = solver.solve(model, tee = True)
print(results['Solver'][0]['Termination condition'].value)

# %% generate dataset
import random
num_data = 10000
data = []
counter = 0
finish = False
while len(data) < num_data:
    # randomly generate parameters
    slope_1 = random.uniform(-2, 0)
    slope_2 = random.uniform(0,2)
    vert_1 = random.uniform(-5, 5)
    vert_2 = random.uniform(-5, 5)
    # create model
    model = ConcreteModel()
    model.x = Var(bounds=(0, 10), domain = NonNegativeReals)
    model.y = Var(bounds=(0, 10), domain = NonNegativeReals)
    # add constraints
    generate_problem(model, slope_1, vert_1, slope_2, vert_2)
    # add objective
    model.obj = Objective(
        expr = model.y + 3*model.x,
        sense = minimize
        )
    # solve and store value
    solver = SolverFactory('ipopt')
    solver.options['mu_strategy'] = 'adaptive'
    solver.options['bound_relax_factor'] = 1e-6
    solver.options['expect_infeasible_problem'] = 'yes'
    solver.options['print_level'] = 0
    results = solver.solve(model)
    if results.solver.termination_condition == TerminationCondition.infeasible:
        counter += 1
    else:
        if len(data) < num_data:
            data.append([slope_1, vert_1, slope_2, vert_2, model.x.value, model.y.value])
        elif finish == False:
            # store dataset
            writer = pd.ExcelWriter('base_file.xlsx')
            data = pd.DataFrame(data)
            data.columns = ["m1", "b1", "m2", "b2", "xopt", "yopt"]
            data.to_excel(writer,  index = False)
            writer.close()    
            finish = True
        counter += 1
    if counter % 100 == 0:
        print(f'counter: {counter} \t data: {len(data)}')



# %% for dataset, select 10 random values   
import re
data_df = pd.read_excel('base_file.xlsx')
data_samples = data_df.sample(n=10)

# write out 
solver = SolverFactory('ipopt')
solver.options["output_file"] = "ipopt_log.txt"
iter_logs = []
for r, row in data_samples.iterrows():
    open("ipopt_log.txt", "w").close()
    model = ConcreteModel()
    model.x = Var(bounds=(0, 10), initialize = 5, domain = NonNegativeReals)
    model.y = Var(bounds=(0, 10),initialize = 5,  domain = NonNegativeReals)
    generate_problem(model, row['m1'], row['b1'], row['m2'], row['b2'])
    # add objective
    model.obj = Objective(
    expr = model.y + 3*model.x,
    sense = minimize
    )
    # solve
    results = solver.solve(model, tee = True)
    # read file 
    with open("ipopt_log.txt") as f:
        log = f.read()
    iters = []
    for line in log.splitlines():
        if re.match(r"\s*\d+\s", line):  # lines starting with iteration number
            x = line.split(' ')
            x = [i for i in x if i not in ['', '-']]
            for n, num in enumerate(x): 
                if 'f' in num: 
                    x[n] = num.split('f')[0]
                if 'h' in num: 
                    x[n] = num.split('h')[0]
            x = [float(i) for i in x]
            iters.append(x)
    iter_logs.append(iters)
    # close file 
    open("ipopt_log.txt", "w").close()





# %% plot important points 
import matplotlib.pyplot as plt 
title_str = 'iter objective inf_pr inf_du lf(mu) ||d|| lg(rg) alpha_du alpha_pr ls'
titles = title_str.split(' ')

# plot against objectives 
title = 'alpha_du'
title_ind = titles.index(title)
for iter in iter_logs: 
    iterations = [x[0] for x in iter]
    other_fact = [x[title_ind] for x in iter]
    plt.plot(iterations, other_fact)
plt.legend()
plt.title(title)
plt.show()
# how many lengths? 
# iterations: 8 - 14 
# inf_pr: 2 - 4
# inf_du: 5 - 13
# alpha_du: 5 - 14
# alpha_pr: 2 
# ||d||: 4 - 8

# # plot primal and dual steps 
# title = 'alpha_du'
# title_ind = titles.index(title)
# d_ind = titles.index('||d||')
# for iter in iter_logs: 
#     iterations = [x[0] for x in iter]
#     other = [x[title_ind] * x[d_ind] for x in iter]
#     plt.plot(iterations, other)
# plt.title(title)
# plt.show()




# %%
