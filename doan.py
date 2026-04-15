import numpy as np
import random
import time
import matplotlib.pyplot as plt

beta = 2
rho = 0.1
q0 = 0.7
tau0 = 0.1
num_ants = 3   

slice_time = 5
Tac = 1
gamma_r = 0.3
vehicle_capacity = 50

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

    best_j = None
    best_val = -1

    for j in F:
        eta = 1 / (dist[i][j] + 1e-6)
        val = pheromone[i][j] * (eta ** beta)

        if val > best_val:
            best_val = val
            best_j = j

    return best_j

def EventsManager(orders, dist):

    Time = 0
    Tco = 30
    Tts = slice_time
    Tnts = Tts   

    PendOrds = [o for o in orders if o["time"] <= 0]
    Committed = []

    pheromone = np.ones((len(dist), len(dist))) * tau0

    vehicle_pos = [0]
    vehicle_cap = [vehicle_capacity]

    while (len(PendOrds) > 0 or Time < Tco):

        if Time > 0:
            start_time = Time + Tac
        else:
            start_time = Time

        StaticOrders = [o for o in PendOrds if o["time"] <= start_time]

        if not StaticOrders:
            Time += Tts
            continue

        # build demand
        demand = [0] * len(dist)
        for o in StaticOrders:
            demand[o["customer"]] = o["demand"]

        BestSol, BestCost, pheromone = ACS(dist, demand, pheromone)

        CommOrds = []

        if BestSol is not None:
            if not CommOrds and StaticOrders:
                CommOrds.append(StaticOrders[0])
            
            arrival = compute_arrival_times(BestSol, dist)

            for o in StaticOrders:
                cust = o["customer"]

                if cust in arrival:
                    processing_time = arrival[cust]

                    
                    if processing_time >= Time + Tnts:
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

        pheromone = pheromone_conservation(pheromone, len(dist))

    if vehicle_pos[0] != 0:
        vehicle_pos[0] = 0

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

    # 🔥 dùng iteration thay vì time
    max_iter = 3

    for _ in range(max_iter):

        for _ in range(num_ants):

            ant = Ant(n, demand)

            # 🔥 chặn vòng lặp vô hạn
            max_steps = n * 2
            steps = 0

            while steps < max_steps:
                steps += 1

                j = select_next(ant, pheromone, dist)

                if j is None:
                    if ant.tour[-1] != 0:
                        ant.return_depot()

                    if not ant.feasible():
                        break
                    else:
                        continue

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
        [0, 2, 9, 10],
        [2, 0, 6, 4],
        [9, 6, 0, 3],
        [10, 4, 3, 0]
    ])

    coords = np.array([
        [0, 0],    
        [2, 3],
        [5, 2],
        [6, 6]
    ])

    orders = [
        {"customer": 1, "demand": 10, "time": 0},
        {"customer": 2, "demand": 20, "time": 10},
        {"customer": 3, "demand": 15, "time": 20}
    ]

    result = EventsManager(orders, dist)

    print("Committed Orders:", result)

# test route từ ACS cuối cùng
    demand = [0]*len(dist)
    for o in result:
        demand[o["customer"]] = o["demand"]

    route, _, _ = ACS(dist, demand, None)

    if route:
        plot_route(route, coords)