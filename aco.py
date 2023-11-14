import numpy as np

from mealpy import FloatVar, ACOR


def objective_function(solution):

    return np.sum(solution**2)


problem_dict = {

    "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),

    "obj_func": objective_function,

    "minmax": "min",

}


model = ACOR.OriginalACOR(epoch=1000, pop_size=50, sample_count = 25, intent_factor = 0.5, zeta = 1.0)

g_best = model.solve(problem_dict)

print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")