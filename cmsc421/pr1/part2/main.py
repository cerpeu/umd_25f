import argparse
import heapq
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.sparse.csgraph import minimum_spanning_tree

def read_matrix(path) -> np.ndarray:
    mat = np.loadtxt(path)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Loaded matrix is not square")

    if np.allclose(mat, mat.T):
        np.fill_diagonal(mat, 0.0)
        return mat.astype(float, copy=False)
    
    if np.allclose(np.tril(mat, -1), 0.0):
        full = mat + mat.T - np.diag(np.diag(mat))
        np.fill_diagonal(full, 0.0)
        return full.astype(float, copy=False)
    
    if np.allclose(np.triu(mat, 1), 0.0):
        full = mat + mat.T - np.diag(np.diag(mat))
        np.fill_diagonal(full, 0.0)
        return full.astype(float, copy=False)
    
    raise ValueError("Matrix is neither symmetric nor purely triangular")

def travel_length(mat: np.ndarray, travel):
    return float(sum(mat[travel[i], travel[i+1]] for i in range(len(travel)-1))
    )
    
def print_n_save(msg, f = None, end= "\n") -> None:
    print(msg, end = end)
    if f is not None:
        f.write(msg + end)



def mst_heuristic(current, visited_mask, dist: np.ndarray):
    n = dist.shape[0]
    nodes = [current] + [v for v in range(n) if ((visited_mask >> v) & 1) == 0]
    if len(nodes) <= 1:
        return 0.0
    
    sub = dist[np.ix_(nodes, nodes)].astype(float, copy=False)
    sub = np.minimum(sub, sub.T)
    mst = minimum_spanning_tree(sub)
    return float(mst.sum())

@dataclass(order=True)
class PQItem:
    f: float
    g: float
    current: int
    visited_mask: int
    path: Tuple[int, ...]

def a_star(dist:np.ndarray,start = 0,time_limit_sec = None):
    n = dist.shape[0]
    if n == 0:
        return []
    w0 = time.perf_counter_ns()

    visited0 = 1 << start
    h0 = mst_heuristic(start, visited0, dist)
    heap: List[PQItem] = [PQItem(f=h0, g= 0.0, current=start, visited_mask=visited0, path=(start,))]
    heapq.heapify(heap)

    best_g: Dict[Tuple[int, int], float] = {(start, visited0): 0.0}
 

    ALL = (1 << n) - 1

    while heap:
        if time_limit_sec is not None and (time.perf_counter_ns() - w0) / 1e9 > time_limit_sec:
            raise TimeoutError("A* time limit exceeded")

        item = heapq.heappop(heap)
        cur, mask, g_cost, path = item.current, item.visited_mask, item.g, item.path

        if mask == ALL and cur == start and len(path) == n + 1:
            return list(path) 

        if mask == ALL and cur != start:
            g2 = g_cost + float(dist[cur, start])
            key = (start, mask)
            old_best = best_g.get(key)
            if old_best is None or g2 < old_best:
                best_g[key] = g2
                heapq.heappush(
                    heap,
                    PQItem(f=g2, g=g2, current=start, visited_mask=mask, path=path + (start,))
                )
            continue

        for nxt in range(n):
            if (mask >> nxt) & 1:
                continue
            g2 = g_cost + float(dist[cur, nxt])
            mask2 = mask | (1 << nxt)
            key = (nxt, mask2)

            h2 = mst_heuristic(nxt, mask2, dist)
            f2 = g2 + h2

            old_best = best_g.get(key)
            if old_best is not None and g2 >= old_best:
                continue
            best_g[key] = g2

            heapq.heappush(heap, PQItem(f=f2, g=g2, current=nxt, visited_mask=mask2, path=path + (nxt,)))

    raise RuntimeError("No solution found (unexpected on complete metric TSP)")        


def main():
    parser = argparse.ArgumentParser(description="A* TSP (NumPy) with SciPy MST heuristic - file input only")
    parser.add_argument("input", type=str, help="Path to adjacency matrix file (read by numpy.loadtxt).")
    parser.add_argument("--start", type=int, default=0, help="Start city index (0-based).")
    parser.add_argument("--time_limit", type=float, default=60.0, help="Wall-clock time limit in seconds.")
    parser.add_argument("--save", type=str, default= "", help="path to save result")
    parser.add_argument("--print-travel", action="store_true", help="print travel route")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"input file not found: {args.input}")
    
    mat = read_matrix(args.input)
    n = mat.shape[0]
    if not (0 <= args.start < n):
        raise SystemExit(f"--start must be in [0, {n-1}], got {args.start}")
    
    wct_0 = time.perf_counter_ns()
    rct_0 = time.process_time_ns()

    
    travel = a_star(mat, start=args.start, time_limit_sec=args.time_limit)
    cost = travel_length(mat, travel)

    wct_1 = time.perf_counter_ns()
    rct_1 = time.process_time_ns()

    f= None
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        f = open(args.save, "w", encoding="utf-8")
    
    wall_time_sec = (wct_1 - wct_0) / 1e9
    cpu_time_sec = (rct_1 - rct_0) / 1e9

    print_n_save(f"runtime(sec): {wall_time_sec:.6f}", f)
    print_n_save(f"cpu_time(sec): {cpu_time_sec:.6f}", f)
    print_n_save(f"cost: {cost:.6f}", f)

    if args.print_travel:
        print_n_save("travel: " + " ".join(map(str, travel)),f)
    if f is not None:
        f.close()

    print(f"{wall_time_sec}, {cpu_time_sec}, {cost}")

if __name__ == "__main__":
    main()


