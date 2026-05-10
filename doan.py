import numpy as np
import random
import time
import matplotlib.pyplot as plt

beta = 1
rho = 0.1
q0 = 0.9
tau0 = 0.1
num_ants = 10

n_ts = 25 
T = 1500
slice_time = T / (6 * n_ts)
Tac = 0
gamma_r = 0.3
vehicle_capacity = 100

def compute_cost(tour, dist):
    return sum(dist[tour[i]][tour[i+1]] for i in range(len(tour)-1))

class Ant:
    def __init__(self, n, demand):
        self.n = n
        self.demand = demand
        self.capacity = vehicle_capacity
        self.remaining = vehicle_capacity
        self.tour = [0]
        self.visited = set([0])

    def feasible(self):
        return [i for i in range(1, self.n)
                if i not in self.visited and self.demand[i] <= self.remaining]

    def visit(self, j):
        self.tour.append(j)
        self.visited.add(j)
        self.remaining -= self.demand[j]

    def return_depot(self):
        self.tour.append(0)
        self.remaining = self.capacity

def select_next(ant, pheromone, dist):
    F = ant.feasible()
    if not F:
        return None

    i = ant.tour[-1]

    if random.random() < q0:
        # exploitation (greedy)
        return max(F, key=lambda j: pheromone[i][j] * (1/(dist[i][j]+1e-6))**beta)
    else:
        # exploration (roulette wheel)
        probs = []
        total = 0

        for j in F:
            val = pheromone[i][j] * (1/(dist[i][j]+1e-6))**beta
            probs.append((j, val))
            total += val

        r = random.random() * total
        s = 0

        for j, val in probs:
            s += val
            if s >= r:
                return j

    return F[0]

def EventsManager(orders, dist):

    Time = 0
    Tco = 0.5 * T
    Tts = slice_time
    Tnts = Tts   
    max_global_steps = 100
    steps = 0

    PendOrds = [o for o in orders if o["time"] <= 0]
    Committed = []

    pheromone = np.ones((len(dist), len(dist))) * tau0

    vehicle_pos = [0]
    vehicle_cap = [vehicle_capacity]

    while (len(PendOrds) > 0 or Time < Tco) and steps < max_global_steps:
        steps += 1

        if Time > 0:
            start_time = Time + Tac
        else:
            start_time = Time

        StaticOrders = [o for o in PendOrds if o["time"] <= start_time]

        if not StaticOrders:
            Time += Tts
            continue

        demand = [0] * len(dist)
        for o in StaticOrders:
            demand[o["customer"]] = o["demand"]

        BestSol, BestCost, pheromone = ACS(dist, demand, pheromone)

        CommOrds = []

        if BestSol is not None:
            arrival = compute_arrival_times(BestSol, dist)

            for o in StaticOrders:
                cust = o["customer"]

                if cust in BestSol:
                    CommOrds.append(o)

        for o in CommOrds:
            if o in PendOrds:
                PendOrds.remove(o)
                Committed.append(o)

        new_orders = [
            o for o in orders
            if Time < o["time"] <= Time + Tnts
        ]

        for o in new_orders:
            if o not in PendOrds and o not in Committed:
                PendOrds.append(o)

        Time += Tts

        if BestSol is not None:
            vehicle_pos[0] = BestSol[-1]
            vehicle_cap[0] = vehicle_capacity - sum(demand[i] for i in BestSol if i != 0)

        pheromone = pheromone_conservation(pheromone, len(dist))

    if vehicle_pos[0] != 0:
        vehicle_pos[0] = 0

    print(f"Thời gian phục vụ={Time}, Đang chờ phục vụ={len(PendOrds)}, Đã phục vụ={len(Committed)}")

    return Committed

def local_update(pheromone, i, j):
    pheromone[i][j] = (1 - rho) * pheromone[i][j] + rho * tau0


def global_update(pheromone, best_tour, best_cost):
    for i in range(len(best_tour)-1):
        a, b = best_tour[i], best_tour[i+1]
        pheromone[a][b] = (1 - rho) * pheromone[a][b] + rho / best_cost 


def ACS(dist, demand, pheromone):

    BestCost = float('inf')
    BestSol = None

    n = len(dist)

    if pheromone is None:
        pheromone = np.ones((n, n)) * tau0

    max_iter = 20

    for _ in range(max_iter):

        for _ in range(num_ants):

            ant = Ant(n, demand)

            max_steps = n * 2
            steps = 0

            while steps < max_steps:
                steps += 1

                j = select_next(ant, pheromone, dist)

                if j is None:
                    break

                i = ant.tour[-1]
                ant.visit(j)

                local_update(pheromone, i, j)

            if ant.tour[-1] != 0:
                ant.return_depot()

            cost = compute_cost(ant.tour, dist)

            if cost < BestCost:
                BestCost = cost
                BestSol = ant.tour.copy()

        if BestSol is not None:
            global_update(pheromone, BestSol, BestCost)

    return BestSol, BestCost, pheromone

def pheromone_conservation(old_pheromone, size):
    new_pheromone = np.ones((size, size)) * tau0

    for i in range(size):
        for j in range(size):
            if i < len(old_pheromone) and j < len(old_pheromone):
                new_pheromone[i][j] = (1 - gamma_r) * old_pheromone[i][j] + gamma_r * tau0

    return new_pheromone


def compute_arrival_times(tour, dist):
    time = 0
    arrival = {}

    for i in range(len(tour)-1):
        a, b = tour[i], tour[i+1]
        time += dist[a][b]
        arrival[b] = time

    return arrival

def GRASP(dist, demand, max_iter=10):

    best_cost = float('inf')
    best_sol = None
    n = len(dist)

    for _ in range(max_iter):
        remaining = vehicle_capacity
        visited = set([0])
        tour = [0]

        while len(visited) < n:
            candidates = []

            for j in range(1, n):
                if j not in visited and demand[j] <= remaining:
                    cost = dist[tour[-1]][j]
                    candidates.append((j, cost))

            if not candidates:
                tour.append(0)
                remaining = vehicle_capacity
                continue

            candidates.sort(key=lambda x: x[1])
            k = min(2, len(candidates))
            chosen = random.choice(candidates[:k])[0]

            tour.append(chosen)
            visited.add(chosen)
            remaining -= demand[chosen]

        if tour[-1] != 0:
            tour.append(0)

        cost = compute_cost(tour, dist)

        if cost < best_cost:
            best_cost = cost
            best_sol = tour.copy()

    return best_sol, best_cost

def evaluate_algorithm(alg_func, dist, demand, runs=5):

    results = []

    for _ in range(runs):
        if alg_func.__name__ == "ACS":
            _, cost, _ = alg_func(dist, demand, None)
        else:
            _, cost = alg_func(dist, demand)

        results.append(cost)

    return {
        "Min": min(results),
        "Max": max(results),
        "Avg": sum(results) / len(results)
    }

def plot_route(tour, coords, title="Route"):
    x = coords[:, 0]
    y = coords[:, 1]

    plt.figure()
    plt.scatter(x, y)

    for i in range(len(coords)):
        plt.text(x[i], y[i], str(i))

    for i in range(len(tour)-1):
        a, b = tour[i], tour[i+1]
        plt.plot([x[a], x[b]], [y[a], y[b]])

    plt.title(title)
    plt.show()

if __name__ == "__main__":

    dist = np.array([
        [0, 4, 8, 6, 7, 3, 9],
        [4, 0, 5, 7, 6, 8, 4],
        [8, 5, 0, 3, 4, 7, 6],
        [6, 7, 3, 0, 5, 4, 8],
        [7, 6, 4, 5, 0, 6, 3],
        [3, 8, 7, 4, 6, 0, 5],
        [9, 4, 6, 8, 3, 5, 0]
    ])

    coords = np.array([
        [0, 0],    
        [2, 3],
        [5, 2],
        [6, 6],
        [8, 3],
        [3, 7],
        [9, 5]
    ])

    orders = [
        {"customer": 1, "demand": 10, "time": 0},
        {"customer": 2, "demand": 15, "time": 5},
        {"customer": 3, "demand": 20, "time": 10},
        {"customer": 4, "demand": 10, "time": 15},
        {"customer": 5, "demand": 25, "time": 20},
        {"customer": 6, "demand": 15, "time": 25}
    ]

    result = EventsManager(orders, dist)

    print("Các đơn hàng đã phục vụ:", result)

    print("{:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Problem",
        "Min", "Max", "Avg",
        "Min", "Max", "Avg"
    ))

    print("{:<12} {:>34} {:>34}".format(
        "",
        "GRASP-DVRP",
        "ACS-DVRP"
    ))


    demand = [0]*len(dist)
    for o in result:
        demand[o["customer"]] = o["demand"]

    route, _, _ = ACS(dist, demand, None)
    
    for test_id in range(1, 6):

        grasp_result = evaluate_algorithm(GRASP, dist, demand, runs=5)
        acs_result = evaluate_algorithm(ACS, dist, demand, runs=5)

        print("{:<12} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
            f"Test-{test_id}",
            grasp_result["Min"],
            grasp_result["Max"],
            grasp_result["Avg"],
            acs_result["Min"],
            acs_result["Max"],
            acs_result["Avg"]
        ))

    if route:
        plot_route(route, coords)
    