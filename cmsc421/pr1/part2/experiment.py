import time
from pathlib import Path
import main as part2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List

plt.rcParams.update({
    "figure.figsize": (9, 5),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.0,
})

MAIN = Path("./main.py")
DATA_DIR = Path("./mats_911")
OUT_DIR = Path("./outputs")

SIZES = [5, 10, 15, 20, 25, 30]
RRNN_K = 3
RRNN_REPEATS = 20

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def read_matrix(n):
    mats = []
    for m in range(0, 10):
        p = DATA_DIR / f"{n}_random_adj_mat_{m}.txt"
        if p.exists():
            mats.append(p)
    return mats

def nn(mat: np.ndarray, start= 0):
    n = mat.shape[0]
    visited = np.zeros(n, dtype=bool)
    travel = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        visited[cur] = True
        dists = np.where(~visited, mat[cur], np.inf)
        nxt = int(np.argmin(dists))
        travel.append(nxt)
        visited[nxt] = True
        cur = nxt
    travel.append(start)
    return travel

def two_opt(mat: np.ndarray, travel):
    n = len(travel) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = travel[i - 1], travel[i]
                x, y = travel[j], travel[(j + 1) % (n + 1)]
                delta = (mat[a, x] + mat[b, y]) - (mat[a, b] + mat[x, y])
                if delta < 0:
                    travel[i:j + 1] = reversed(travel[i:j + 1])
                    improved = True
                    break
            if improved:
                break
    return travel

def nn_two_opt(mat: np.ndarray, start=0):
    return two_opt(mat, nn(mat, start))

def rrnn(mat: np.ndarray, k = RRNN_K, num_repeats = RRNN_REPEATS, seed=None):
    n = mat.shape[0]
    rng = np.random.default_rng(seed)
    best_travel, best_cost = None, np.inf
    for _ in range(num_repeats):
        start = int(rng.integers(0, n))
        visited = np.zeros(n, dtype=bool)
        travel = [start]
        visited[start] = True
        cur = start
        for _ in range(n - 1):
            dists = np.where(~visited, mat[cur], np.inf)
            dists[cur] = np.inf
            cand = np.where(np.isfinite(dists))[0]
            num_cand = min(k, cand.size)
            part = np.argpartition(dists[cand], num_cand - 1)[:num_cand]
            k_cands = cand[part]
            nxt = int(rng.choice(k_cands))
            travel.append(nxt)
            visited[nxt] = True
            cur = nxt
        travel.append(start)
        travel = two_opt(mat, travel)
        cost = float(sum(mat[travel[i], travel[i + 1]] for i in range(len(travel) - 1)))
        if cost < best_cost:
            best_cost = cost
            best_travel = travel
    return best_travel

def time_inproc(matrix_path, run, *, cpu_zero_runs= 5):
    mat = part2.read_matrix(str(matrix_path))

    t0, c0 = time.perf_counter_ns(), time.process_time_ns()
    tour = run(mat) 
    t1, c1 = time.perf_counter_ns(), time.process_time_ns()
    wall = (t1 - t0) / 1e9
    cpu  = (c1 - c0) / 1e9

    if cpu == 0.0 and cpu_zero_runs > 1:
        t0, c0 = time.perf_counter_ns(), time.process_time_ns()
        for _ in range(cpu_zero_runs):
            _ = run(mat)
        t1, c1 = time.perf_counter_ns(), time.process_time_ns()
        wall = (t1 - t0) / 1e9 / cpu_zero_runs
        cpu  = (c1 - c0) / 1e9 / cpu_zero_runs

    cost = float(part2.travel_length(mat, tour))
    return wall, cpu, cost


def run_astar_nodes(matrix_path, start= 0, time_limit_sec = 60.0):
    import main as part2
    mat = part2.read_matrix(str(matrix_path))

    orig_mst = part2.mst_heuristic
    counter = {"cnt": 0}
    def counting_mst(current, visited_mask, dist):
        counter["cnt"] += 1
        return orig_mst(current, visited_mask, dist)

    part2.mst_heuristic = counting_mst
    try:
        w0, c0 = time.perf_counter_ns(), time.process_time_ns()
        tour = part2.a_star(mat, start=start, time_limit_sec=time_limit_sec)
        w1, c1 = time.perf_counter_ns(), time.process_time_ns()
        wall, cpu = (w1 - w0) / 1e9, (c1 - c0) / 1e9
        nodes = int(counter["cnt"])
    finally:
        part2.mst_heuristic = orig_mst

    cost = float(part2.travel_length(mat, tour))

    if cpu == 0.0:
        runs = 5
        w0, c0 = time.perf_counter_ns(), time.process_time_ns()
        for _ in range(runs):
            _ = part2.a_star(mat, start=start, time_limit_sec=time_limit_sec)
        w1, c1 = time.perf_counter_ns(), time.process_time_ns()
        wall, cpu = (w1 - w0) / 1e9 / runs, (c1 - c0) / 1e9 / runs

    return wall, cpu, cost, nodes

def compare():

    ratio_wall = {"NN": defaultdict(list), "NN2": defaultdict(list), "RR": defaultdict(list)}
    ratio_cpu  = {"NN": defaultdict(list), "NN2": defaultdict(list), "RR": defaultdict(list)}
    ratio_cost = {"NN": defaultdict(list), "NN2": defaultdict(list), "RR": defaultdict(list)}
    nodes_a    = defaultdict(list) 

    STOP_AFTER_CONSEC_TIMEOUTS = 5
    consec_timeouts = 0

    for n in SIZES:
        mats = read_matrix(n)
        if not mats:
            continue

        any_astar_ok = False

        for mat in mats:
            print(f"[run] n={n}  file={mat.name}  (A* first)")
            try:
                w_a, c_a, sc_a, nd = run_astar_nodes(mat, start=0, time_limit_sec=60.0)
                any_astar_ok = True
                consec_timeouts = 0
            except TimeoutError:
                print(f"[timeout] A* at n={n}  file={mat.name}  (skipping this instance)")
                consec_timeouts += 1
                if consec_timeouts >= STOP_AFTER_CONSEC_TIMEOUTS:
                    print(f"[stop] {STOP_AFTER_CONSEC_TIMEOUTS} consecutive A* timeouts → stop larger n.")
                    break
                continue

            print(f"[run] n={n}  file={mat.name}  (NN / NN+2Opt / RRNN)")
            w_nn,  c_nn,  sc_nn  = time_inproc(mat, lambda adj: nn(adj, start=0))
            w_nn2, c_nn2, sc_nn2 = time_inproc(mat, lambda adj: nn_two_opt(adj, start=0))
            w_rr,  c_rr,  sc_rr  = time_inproc(mat, lambda adj: rrnn(adj, k=RRNN_K, num_repeats=RRNN_REPEATS, seed=12345))

            if not np.isclose(w_a, 0.0):
                ratio_wall["NN"][n].append(w_nn / w_a)
                ratio_wall["NN2"][n].append(w_nn2 / w_a)
                ratio_wall["RR"][n].append(w_rr / w_a)
            if not np.isclose(c_a, 0.0):
                ratio_cpu["NN"][n].append(c_nn / c_a)
                ratio_cpu["NN2"][n].append(c_nn2 / c_a)
                ratio_cpu["RR"][n].append(c_rr / c_a)
            if not np.isclose(sc_a, 0.0):
                ratio_cost["NN"][n].append(sc_nn / sc_a)
                ratio_cost["NN2"][n].append(sc_nn2 / sc_a)
                ratio_cost["RR"][n].append(sc_rr / sc_a)

            nodes_a[n].append(nd)

        if consec_timeouts >= STOP_AFTER_CONSEC_TIMEOUTS:
            break
        if not any_astar_ok:
            break

    ns = [n for n in SIZES if n in nodes_a and len(nodes_a[n]) > 0]
    if not ns:
        print("[warn] No successful A* runs, skipping plots.")
        return

    def med_list(dct, key):
        vals = dct.get(key, [])
        return float(np.median(vals)) if vals else float("nan")

    mw_nn, mw_nn2, mw_rr = [], [], []
    mc_nn, mc_nn2, mc_rr = [], [], []
    ms_nn, ms_nn2, ms_rr = [], [], []
    mn_a = []

    for n in ns:
        mw_nn.append( med_list(ratio_wall["NN"],  n) )
        mw_nn2.append(med_list(ratio_wall["NN2"], n))
        mw_rr.append( med_list(ratio_wall["RR"],  n) )

        mc_nn.append( med_list(ratio_cpu["NN"],  n) )
        mc_nn2.append(med_list(ratio_cpu["NN2"], n))
        mc_rr.append( med_list(ratio_cpu["RR"],  n) )

        ms_nn.append( med_list(ratio_cost["NN"],  n) )
        ms_nn2.append(med_list(ratio_cost["NN2"], n))
        ms_rr.append( med_list(ratio_cost["RR"],  n) )

        mn_a.append( float(np.median(nodes_a[n])) )

    ensure_dir(OUT_DIR)

    import csv
    with open(OUT_DIR / "part2_medians.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n",
                    "wall_ratio_nn","wall_ratio_nn2","wall_ratio_rr",
                    "cpu_ratio_nn","cpu_ratio_nn2","cpu_ratio_rr",
                    "cost_ratio_nn","cost_ratio_nn2","cost_ratio_rr",
                    "nodes_a_median"])
        for i, n in enumerate(ns):
            w.writerow([n,
                        mw_nn[i], mw_nn2[i], mw_rr[i],
                        mc_nn[i], mc_nn2[i], mc_rr[i],
                        ms_nn[i], ms_nn2[i], ms_rr[i],
                        mn_a[i]])
    print(f"[saved] {OUT_DIR / 'part2_medians.csv'}")

    plt.figure()
    ax = plt.gca()
    ax.plot(ns, mw_nn,  "o-", label="NN / A*")
    ax.plot(ns, mw_nn2, "o-", label="NN+2opt / A*")
    ax.plot(ns, mw_rr,  "o-", label="RRNN / A*")
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color="#999999")
    ax.set_xlabel("Number of Cities (n)")
    ax.set_ylabel("Wall time ratio (algo / A*)")
    ax.set_title("Part II: Wall time ratio vs A*")
    ax2 = ax.twinx()
    ax2.plot(ns, mn_a, "--x", color="#666666", linewidth=1.2, label="A* nodes expanded (median)")
    ax2.set_ylabel("A* nodes expanded")
    l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part2_ratio_wall.png"); plt.close()

    plt.figure()
    ax = plt.gca()
    ax.plot(ns, mc_nn,  "o-", label="NN / A*")
    ax.plot(ns, mc_nn2, "o-", label="NN+2opt / A*")
    ax.plot(ns, mc_rr,  "o-", label="RRNN / A*")
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color="#999999")
    ax.set_xlabel("Number of Cities (n)")
    ax.set_ylabel("CPU time ratio (algo / A*)")
    ax.set_title("Part II: CPU time ratio vs A*")
    ax2 = ax.twinx()
    ax2.plot(ns, mn_a, "--x", color="#666666", linewidth=1.2, label="A* nodes expanded (median)")
    ax2.set_ylabel("A* nodes expanded")
    l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part2_ratio_cpu.png"); plt.close()

    plt.figure()
    ax = plt.gca()
    ax.plot(ns, ms_nn,  "o-", label="NN / A*")
    ax.plot(ns, ms_nn2, "o-", label="NN+2opt / A*")
    ax.plot(ns, ms_rr,  "o-", label="RRNN / A*")
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color="#999999")
    ax.set_xlabel("Number of Cities (n)")
    ax.set_ylabel("Cost ratio (algo / A*)")
    ax.set_title("Part II: Cost ratio vs A*")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part2_ratio_cost.png"); plt.close()

    plt.figure()
    plt.plot(ns, mn_a, "o-")
    plt.xlabel("Number of Cities (n)")
    plt.ylabel("A* nodes expanded (median)")
    plt.title("Part II: A* nodes expanded vs n")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part2_nodes_expanded.png"); plt.close()

    print("[saved] Part II plots in", OUT_DIR.resolve())

def main():
    ensure_dir(OUT_DIR)
    print("== Ratios vs A* (with nodes expanded) ==")
    compare()
    print("Done, Figures saved in:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()