import numpy as np
import heapq
import time as _time

class RunTrace:
    def __init__(self, best_so_far):
        self.best_so_far = best_so_far

class SolveResult:
    def __init__(self, best_cost, best_route, trace=None):
        self.best_cost = float(best_cost)
        self.best_route = list(best_route)
        self.trace = trace

def rng(seed):
    return np.random.default_rng(seed)

def travel_cost(dist, travel_route):
    n = len(travel_route)
    if n == 0:
        return 0.0
    c = 0.0
    for i in range(n):
        a = travel_route[i]
        b = travel_route[(i + 1) % n]
        c += dist[a, b]
    return float(c)

def swap_delta(dist, travel_route, i, j):
    n = len(travel_route)
    if i == j:
        return 0.0
    if j < i:
        i, j = j, i
    a = travel_route[(i-1) % n]
    b = travel_route[i]
    c = travel_route[(i+1) % n]
    d = travel_route[(j-1) % n]
    e = travel_route[j]
    f = travel_route[(j+1) % n]
    if j == i + 1 or (i == 0 and j == n-1):
        return (dist[a, e] + dist[e, b] + dist[b, f]) - (dist[a, b] + dist[b, e] + dist[e, f])
    else:
        return (dist[a, e] + dist[e, c] + dist[d, b] + dist[b, f]) - (dist[a, b] + dist[b, c] + dist[d, e] + dist[e, f])

def swap(travel_route, i, j):
    travel_route[i], travel_route[j] = travel_route[j], travel_route[i]

def start_zero(travel_route):
    if not travel_route:
        return travel_route
    idx0 = travel_route.index(0)
    if idx0 == 0:
        return travel_route
    return travel_route[idx0:] + travel_route[:idx0]


def hill_climbing(dist, num_restarts, max_iters_per_restart=5000, seed=None, record_trace=True, no_improve_limit=200):
    n = dist.shape[0]
    random = rng(seed)
    global_best_cost = float('inf')
    global_best_route = []
    trace_list = []

    for _ in range(num_restarts):
        travel_route = list(range(n))
        random.shuffle(travel_route)
        travel_route = start_zero(travel_route)
        cost = travel_cost(dist, travel_route)
        best_cost = cost
        best_route = travel_route.copy()
        no_improve = 0

        for _ in range(max_iters_per_restart):
            i = int(random.integers(1, n))
            j = int(random.integers(1, n))
            if i == j: 
                continue
            delta = swap_delta(dist, travel_route, i, j)
            if delta < 0:
                swap(travel_route, i, j)
                cost += float(delta)
                if cost < best_cost:
                    best_cost = cost
                    best_route = travel_route.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            if record_trace:
                trace_list.append(min(global_best_cost, best_cost))
            if no_improve >= no_improve_limit:
                break

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_route = best_route

    return SolveResult(global_best_cost, global_best_route, RunTrace(trace_list) if record_trace else None)

def simulated_annealing(dist, alpha, initial_temperature, max_iterations, seed=None, record_trace=True, no_improve_limit=1000):
    n = dist.shape[0]
    random = rng(seed)
    travel_route = list(range(n))
    random.shuffle(travel_route)
    travel_route= start_zero(travel_route)
    cost = travel_cost(dist, travel_route)
    best_cost = cost
    best_route = travel_route.copy()
    T = float(initial_temperature)
    no_improve = 0
    trace_list = []

    for _ in range(max_iterations):
        i = int(random.integers(1, n))
        j = int(random.integers(1, n))
        if i == j:
            continue
        delta = swap_delta(dist, travel_route, i, j)
        accept = (delta < 0) or (random.random() < np.exp(-delta / max(T, 1e-9)))
        if accept:
            swap(travel_route, i, j)
            cost += float(delta)
            if cost < best_cost:
                best_cost = cost
                best_route = travel_route.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        if record_trace:
            trace_list.append(best_cost)
        T *= alpha
        if no_improve >= no_improve_limit or T < 1e-6:
            break

    return SolveResult(best_cost, best_route, RunTrace(trace_list) if record_trace else None)

def ga_select(rng, pop, scores, k=3):
    n = len(pop)
    idxs = rng.integers(0, n, size=k)
    best_idx = int(idxs[0])
    best_score = scores[best_idx]
    for idx in idxs[1:]:
        s = scores[int(idx)]
        if s < best_score:
            best_score = s
            best_idx = int(idx)
    return best_idx

def ga_crossover(rng, p1, p2):
    n = len(p1)
    a, b = sorted(rng.integers(1, n, size=2)) 
    child = [-1] * n
    child[0] = 0
    child[a:b] = p1[a:b]
    used = set(child[a:b])
    pos = b
    for x in p2:
        if x == 0 or x in used:
            continue
        if pos == n: pos = 1
        child[pos] = x
        pos += 1
    return child

def mutate_individual_swap(random, travel_route):
    n = len(travel_route)
    i = int(random.integers(1, n))
    j = int(random.integers(1, n))
    while j == i: 
        j = int(random.integers(1, n))
    return i, j

def genetic_algorithm(dist, mutation_chance, population_size, num_generations, seed=None, record_trace=True, stagnate_limit=10):
    n = dist.shape[0]
    random = rng(seed)

    population = []
    scores = []
    base = list(range(n))
    for _ in range(population_size):
        travel_route = base.copy()
        random.shuffle(travel_route)
        travel_route = start_zero(travel_route)
        population.append(travel_route)
        scores.append(travel_cost(dist, travel_route))

    best_idx = int(np.argmin(scores))
    best_cost = scores[best_idx]
    best_route = population[best_idx].copy()
    trace_list = []
    stagnate = 0

    for _ in range(num_generations):
        children = []
        child_scores = []
        for _ in range(population_size):
            i = ga_select(random, population, scores, k=3)
            j = ga_select(random, population, scores, k=3)
            while j == i: 
                j = ga_select(random, population, scores, k=3)
            p1, p2 = population[i], population[j]
            child = ga_crossover(random, p1, p2)
            cost = travel_cost(dist, child)
            if random.random() < mutation_chance:
                a, b = mutate_individual_swap(random, child)
                delta = swap_delta(dist, child, a, b)
                swap(child, a, b)
                cost += float(delta)
            children.append(child)
            child_scores.append(cost)

        merged = population + children
        merged_scores = scores + child_scores
        order = np.argsort(merged_scores)[:population_size]
        population = [merged[idx] for idx in order]
        scores     = [merged_scores[idx] for idx in order]

        gen_best_idx = int(np.argmin(scores))
        gen_best = scores[gen_best_idx]
        if gen_best < best_cost:
            best_cost = gen_best
            best_route = population[gen_best_idx].copy()
            stagnate = 0
        else:
            stagnate += 1
        if record_trace:
            trace_list.append(best_cost)
        if stagnate >= stagnate_limit:
            break

    return SolveResult(best_cost, best_route, RunTrace(trace_list) if record_trace else None)

def prim_mst(dist, nodes):
    if not nodes:
        return 0.0
    m = len(nodes)
    in_mst = [False]*m
    key = [float('inf')]*m
    key[0] = 0.0
    total = 0.0
    for _ in range(m):
        u = -1
        minv = float('inf')
        for i in range(m):
            if not in_mst[i] and key[i] < minv:
                minv = key[i]
                u = i
        in_mst[u] = True
        total += key[u]
        for v in range(m):
            if not in_mst[v]:
                w = dist[nodes[u], nodes[v]]
                if w < key[v]:
                    key[v] = w
    return total

def mst_heuristic(current, visited_mask, dist, start=0):
    n = dist.shape[0]
    unvisited = [i for i in range(n) if not (visited_mask >> i) & 1]
    if not unvisited:
        return dist[current, start]
    h = prim_mst(dist, unvisited)
    h += float(np.min(dist[current, unvisited]))
    h += float(np.min(dist[unvisited, start]))
    return float(h)

def a_star(dist, start=0, time_limit_sec=None):
    n = dist.shape[0]
    if n == 0:
        return []

    start_mask = 1 << start
    heap = []
    h0 = mst_heuristic(start, start_mask, dist, start)
    heapq.heappush(heap, (h0, 0.0, start, start_mask, (start,)))  # (f, g, current, mask, path)
    best_g = {(start, start_mask): 0.0}

    t0 = _time.perf_counter()
    full_mask = (1 << n) - 1

    while heap:
        if time_limit_sec is not None and (_time.perf_counter() - t0) > time_limit_sec:
            break

        f, g, cur, mask, path = heapq.heappop(heap)

        if mask == full_mask:
            return list(path)

        for nxt in range(n):
            if (mask >> nxt) & 1:
                continue
            g2 = g + dist[cur, nxt]
            mask2 = mask | (1 << nxt)
            key = (nxt, mask2)
            if key in best_g and g2 >= best_g[key]:
                continue
            best_g[key] = g2

            if mask2 == full_mask:
                pass

            h = mst_heuristic(nxt, mask2, dist, start)
            heapq.heappush(heap, (g2 + h, g2, nxt, mask2, path + (nxt,)))

    return list(range(n))
