import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
import numpy as np


GLOBAL_ENV = gp.Env(empty=True)
GLOBAL_ENV.setParam("LogToConsole", 0)  # silence repeated log messages
GLOBAL_ENV.start()


def tsp_optimal_length(distances: np.ndarray):
    cities = list(range(len(distances)))

    def solve_tsp(optimize_for="min"):
        dist = {(i, j): distances[i, j] for i in cities for j in cities if i < j}

        m = gp.Model(env=GLOBAL_ENV)
        m.Params.LogToConsole = 0  # silence output

        if optimize_for == "max":
            dist_obj = {k: -v for k, v in dist.items()}
        else:
            dist_obj = dist

        vars = m.addVars(dist.keys(), obj=dist_obj, vtype=GRB.BINARY, name='x')

        vars.update({(j, i): vars[i, j] for i, j in dist.keys()})

        m.addConstrs(vars.sum(c, '*') == 2 for c in cities)

        def subtourelim(model, where):
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(model._vars)
                selected = gp.tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
                tour = subtour(selected)
                if len(tour) < len(cities):
                    model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2)) <= len(tour) - 1)

        def subtour(edges):
            unvisited = cities[:]
            cycle = cities[:]  # dummy
            while unvisited:
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
                if len(thiscycle) <= len(cycle):
                    cycle = thiscycle
            return cycle

        m._vars = vars
        m.Params.lazyConstraints = 1
        m.optimize(subtourelim)

        if optimize_for == "max":
            return -m.objVal
        else:
            return m.objVal

    l_star = solve_tsp("min")
    l_worst = solve_tsp("max")

    return l_star, l_worst


if __name__ == "__main__":
    D = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
    ])
    l_star, l_worst = tsp_optimal_length(D)
    print("Optimal:", l_star)
    print("Worst:", l_worst)
