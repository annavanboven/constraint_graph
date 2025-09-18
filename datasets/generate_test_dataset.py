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
model.x = Var(bounds=(0, 10), domain = NonNegativeReals)
model.y = Var(bounds=(0, 10), domain = NonNegativeReals)
generate_problem(model, 3, -2, 4, -3)
# add objective
model.obj = Objective(
expr = model.y + 3*model.x,
sense = minimize
)
solver = SolverFactory('ipopt')
results = solver.solve(model,)
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



# %%
