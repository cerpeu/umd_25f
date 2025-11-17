import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import main as algo
from pathlib import Path
from collections import defaultdict
DATA_DIR = Path('./mats_911')
OUT_DIR = Path('./outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)
SIZES = [5, 10, 15, 20, 25, 30]
HC_NUM_RESTARTS_GRID = [5, 10, 20, 40, 80]
HC_MAX_ITERS = 5000
SA_ALPHA_GRID = [0.99, 0.993, 0.995, 0.996, 0.997]
SA_MAX_ITERS = 5000
SA_TEMP = 100.0
GA_MUTATION_GRID = [0.02, 0.05, 0.1, 0.15, 0.2]
GA_POP_GRID = [10, 15, 20, 25, 30]
GA_GENS = 50
TRAJ_SA_REPEATS = 3
TRAJ_GA_REPEATS = 3
TIME_LIMIT_ASTAR = 60.0

def read_matrix(path):
    with path.open('r') as f:
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split('[,\\s]+', line)
            rows.append([float(x) for x in parts])
    return np.array(rows, dtype=float)

def instances_size():
    by_size = defaultdict(list)
    for n in SIZES:
        files = sorted(DATA_DIR.glob(f'{n}_random_adj_mat_*.txt'))
        for p in files:
            dist = read_matrix(p)
            by_size[n].append((p.stem, dist))
    return by_size

def median_ignore_nan(x):
    x = np.array(x, dtype=float)
    return float(np.nanmedian(x)) if x.size else float('nan')

def ensure_len(arr, L):
    if len(arr) >= L:
        return np.array(arr[:L], dtype=float)
    if not arr:
        return np.full(L, np.nan, dtype=float)
    return np.array(list(arr) + [arr[-1]] * (L - len(arr)), dtype=float)

def median_trace(traces, out_path, title):
    if not traces:
        return
    L = max((len(t) for t in traces))
    mat = np.vstack([ensure_len(t, L) for t in traces])
    y = np.nanmedian(mat, axis=0)
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, L + 1), y, marker='o')
    plt.title(title)
    plt.xlabel('Iteration / Generation')
    plt.ylabel('Best-so-far cost (median)')
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

def plot_overall_curve(xgrid, meds, xlabel, title, out_path):
    plt.figure(figsize=(8.5, 5))
    plt.plot(xgrid, meds, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Overall median cost')
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

def mix(x):
    x &= 4294967295
    x ^= x >> 16
    x = x * 2146121005 & 4294967295
    x ^= x >> 15
    x = x * 2221713035 & 4294967295
    x ^= x >> 16
    return x

def make_seed(tag, n, inst_idx, gi, p1, p2=0, rep=0):
    acc = 19088743
    for ch in tag:
        acc = acc * 131 + ord(ch) & 4294967295
    acc ^= n * 1000003 & 4294967295
    acc ^= inst_idx * 9176 & 4294967295
    acc ^= gi * 2713 & 4294967295
    acc ^= p1 * 911 & 4294967295
    acc ^= p2 * 3571 & 4294967295
    acc ^= rep * 7919 & 4294967295
    return mix(acc)

def extract_trace(solve_result):
    tr = getattr(solve_result, 'trace', None)
    return getattr(tr, 'best_so_far', None) if tr is not None else None

def run_hc(dist, restarts, record_trace, seed=None):
    w0 = time.perf_counter()
    c0 = time.process_time()
    sr = algo.hill_climbing(dist, num_restarts=restarts, max_iters_per_restart=HC_MAX_ITERS, seed=seed, record_trace=record_trace)
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    return {'best_cost': float(getattr(sr, 'best_cost', float('nan'))), 'wall': wall, 'cpu': cpu, 'trace': extract_trace(sr)}

def run_sa(dist, alpha, record_trace, seed=None):
    w0 = time.perf_counter()
    c0 = time.process_time()
    sr = algo.simulated_annealing(dist, alpha=alpha, initial_temperature=SA_TEMP, max_iterations=SA_MAX_ITERS, seed=seed, record_trace=record_trace)
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    return {'best_cost': float(getattr(sr, 'best_cost', float('nan'))), 'wall': wall, 'cpu': cpu, 'trace': extract_trace(sr)}

def run_ga(dist, mut, pop, record_trace, seed=None):
    w0 = time.perf_counter()
    c0 = time.process_time()
    sr = algo.genetic_algorithm(dist, mutation_chance=mut, population_size=pop, num_generations=GA_GENS, seed=seed, record_trace=record_trace)
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    return {'best_cost': float(getattr(sr, 'best_cost', float('nan'))), 'wall': wall, 'cpu': cpu, 'trace': extract_trace(sr)}

def run_astar(dist, time_limit_sec=TIME_LIMIT_ASTAR):
    w0 = time.perf_counter()
    c0 = time.process_time()
    tour = algo.a_star(dist, time_limit_sec=time_limit_sec)
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    cost = algo.travel_cost(dist, tour) if tour else float('nan')
    return {'best_cost': float(cost), 'wall': wall, 'cpu': cpu}

def hc_sweep(instances):
    hc_overall_meds = []
    for gi, nr in enumerate(HC_NUM_RESTARTS_GRID):
        per_size_meds = []
        for n in SIZES:
            costs = []
            for inst_idx, (name, dist) in enumerate(instances.get(n, [])):
                seed = make_seed('HC', n, inst_idx, gi, nr)
                res = run_hc(dist, restarts=nr, record_trace=False, seed=seed)
                costs.append(res['best_cost'])
            per_size_meds.append(median_ignore_nan(costs))
        hc_overall_meds.append(median_ignore_nan(per_size_meds))
    plot_overall_curve(HC_NUM_RESTARTS_GRID, hc_overall_meds, xlabel='HC restarts', title='HC sweep – overall', out_path=str(OUT_DIR / 'hc_sweep_overall.png'))
    best_idx = int(np.nanargmin(hc_overall_meds))
    return HC_NUM_RESTARTS_GRID[best_idx]

def sa_sweep(instances):
    sa_overall_meds = []
    for gi, alpha in enumerate(SA_ALPHA_GRID):
        per_size_meds = []
        alpha_i = int(round(alpha * 1000))
        for n in SIZES:
            costs = []
            for inst_idx, (name, dist) in enumerate(instances.get(n, [])):
                seed = make_seed('SA', n, inst_idx, gi, alpha_i)
                res = run_sa(dist, alpha=alpha, record_trace=False, seed=seed)
                costs.append(res['best_cost'])
            per_size_meds.append(median_ignore_nan(costs))
        sa_overall_meds.append(median_ignore_nan(per_size_meds))
    plot_overall_curve(SA_ALPHA_GRID, sa_overall_meds, xlabel='SA alpha', title='SA sweep – overall', out_path=str(OUT_DIR / 'sa_sweep_overall.png'))
    best_idx = int(np.nanargmin(sa_overall_meds))
    return SA_ALPHA_GRID[best_idx]

def ga_sweep_mut(instances, ga_fix_pop):
    ga_mut_overall_meds = []
    for gi, mut in enumerate(GA_MUTATION_GRID):
        per_size_meds = []
        mut_i = int(round(mut * 10000))
        for n in SIZES:
            costs = []
            for inst_idx, (name, dist) in enumerate(instances.get(n, [])):
                seed = make_seed('GA_MUT', n, inst_idx, gi, mut_i, ga_fix_pop)
                res = run_ga(dist, mut=mut, pop=ga_fix_pop, record_trace=False, seed=seed)
                costs.append(res['best_cost'])
            per_size_meds.append(median_ignore_nan(costs))
        ga_mut_overall_meds.append(median_ignore_nan(per_size_meds))
    plot_overall_curve(GA_MUTATION_GRID, ga_mut_overall_meds, xlabel='GA mutation rate', title='GA mutation sweep – overall', out_path=str(OUT_DIR / 'ga_mut_sweep_overall.png'))
    best_idx = int(np.nanargmin(ga_mut_overall_meds))
    return GA_MUTATION_GRID[best_idx]

def ga_sweep_pop(instances, ga_fix_mut):
    ga_pop_overall_meds = []
    mut_i = int(round(ga_fix_mut * 10000))
    for gi, pop in enumerate(GA_POP_GRID):
        per_size_meds = []
        for n in SIZES:
            costs = []
            for inst_idx, (name, dist) in enumerate(instances.get(n, [])):
                seed = make_seed('GA_POP', n, inst_idx, gi, pop, mut_i)
                res = run_ga(dist, mut=ga_fix_mut, pop=pop, record_trace=False, seed=seed)
                costs.append(res['best_cost'])
            per_size_meds.append(median_ignore_nan(costs))
        ga_pop_overall_meds.append(median_ignore_nan(per_size_meds))
    plot_overall_curve(GA_POP_GRID, ga_pop_overall_meds, xlabel='GA population size', title='GA population sweep – overall', out_path=str(OUT_DIR / 'ga_pop_sweep_overall.png'))
    best_idx = int(np.nanargmin(ga_pop_overall_meds))
    return GA_POP_GRID[best_idx]

def best_repeats(traces):
    if not traces:
        return []
    finals = [t[-1] if len(t) > 0 else float('inf') for t in traces]
    idx = int(np.nanargmin(finals))
    return list(traces[idx])

def representative_record(instances, hc_best_nr, sa_best_alpha, ga_best_mut, ga_best_pop):
    print('[INFO] Phase 3 – representative trajectories')
    for n in SIZES:
        traces = []
        for name, dist in instances.get(n, []):
            seed = make_seed('HC_TRAJ', n, 0, 0, hc_best_nr)
            res = run_hc(dist, restarts=hc_best_nr, record_trace=True, seed=seed)
            if res['trace'] is not None:
                traces.append(list(res['trace']))
        if traces:
            median_trace(traces, str(OUT_DIR / f'hc_traj_n{n}.png'), title=f'HC trajectory (n={n}, restarts={hc_best_nr})')
    for n in SIZES:
        inst_traces = []
        alpha_i = int(round(sa_best_alpha * 1000))
        for inst_idx, (name, dist) in enumerate(instances.get(n, [])):
            rep_traces = []
            for rep in range(TRAJ_SA_REPEATS):
                seed = make_seed('SA_TRAJ', n, inst_idx, 0, alpha_i, 0, rep)
                res = run_sa(dist, alpha=sa_best_alpha, record_trace=True, seed=seed)
                if res['trace'] is not None:
                    rep_traces.append(list(res['trace']))
            if rep_traces:
                inst_traces.append(best_repeats(rep_traces))
        if inst_traces:
            median_trace(inst_traces, str(OUT_DIR / f'sa_traj_n{n}.png'), title=f'SA trajectory (n={n}, alpha={sa_best_alpha})')
    for n in SIZES:
        inst_traces = []
        mut_i = int(round(ga_best_mut * 10000))
        for inst_idx, (name, dist) in enumerate(instances.get(n, [])):
            rep_traces = []
            for rep in range(TRAJ_GA_REPEATS):
                seed = make_seed('GA_TRAJ', n, inst_idx, 0, ga_best_pop, mut_i, rep)
                res = run_ga(dist, mut=ga_best_mut, pop=ga_best_pop, record_trace=True, seed=seed)
                if res['trace'] is not None:
                    rep_traces.append(list(res['trace']))
            if rep_traces:
                inst_traces.append(best_repeats(rep_traces))
        if inst_traces:
            median_trace(inst_traces, str(OUT_DIR / f'ga_traj_n{n}.png'), title=f'GA trajectory (n={n}, mut={ga_best_mut}, pop={ga_best_pop})')

def plot(med_table):
    algos = ['HC', 'SA', 'GA']
    labels = {'wall': 'Runtime ratio (wall / A*)', 'cpu': 'CPU time ratio (cpu / A*)', 'score': 'Cost ratio (cost / A*)'}
    files = {'wall': 'part3_ratio_wall.png', 'cpu': 'part3_ratio_cpu.png', 'score': 'part3_ratio_cost.png'}
    for metric in ['wall', 'cpu', 'score']:
        xs = []
        ys = {a: [] for a in algos}
        for n in SIZES:
            row = med_table.get(n, {})
            if 'A*' not in row:
                continue
            base = row['A*'].get(metric, np.nan)
            if not np.isfinite(base) or base == 0:
                continue
            xs.append(n)
            for a in algos:
                val = row.get(a, {}).get(metric, np.nan)
                ys[a].append(val / base if np.isfinite(val) else np.nan)
        if not xs:
            continue
        plt.figure(figsize=(8.8, 5.2))
        for a in algos:
            if ys[a]:
                plt.plot(xs, ys[a], marker='o', label=a)
        plt.title(labels[metric])
        plt.xlabel('Number of cities (n)')
        plt.ylabel(labels[metric])
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / files[metric]), dpi=140)
        plt.close()

def summary(instances, hc_best_nr, sa_best_alpha, ga_best_mut, ga_best_pop):
    print('[INFO] Phase 4 – summaries, A* baseline & ratios (A* cap = 60s)')
    med_table = {n: {} for n in SIZES}
    for n in SIZES:
        scores, walls, cpus = ([], [], [])
        for name, dist in instances.get(n, []):
            seed = make_seed('HC_SUM', n, 0, 0, hc_best_nr)
            res = run_hc(dist, restarts=hc_best_nr, record_trace=False, seed=seed)
            scores.append(res['best_cost'])
            walls.append(res['wall'])
            cpus.append(res['cpu'])
        med_table[n]['HC'] = {'score': median_ignore_nan(scores), 'wall': median_ignore_nan(walls), 'cpu': median_ignore_nan(cpus)}
    for n in SIZES:
        scores, walls, cpus = ([], [], [])
        alpha_i = int(round(sa_best_alpha * 1000))
        for name, dist in instances.get(n, []):
            seed = make_seed('SA_SUM', n, 0, 0, alpha_i)
            res = run_sa(dist, alpha=sa_best_alpha, record_trace=False, seed=seed)
            scores.append(res['best_cost'])
            walls.append(res['wall'])
            cpus.append(res['cpu'])
        med_table[n]['SA'] = {'score': median_ignore_nan(scores), 'wall': median_ignore_nan(walls), 'cpu': median_ignore_nan(cpus)}
    for n in SIZES:
        scores, walls, cpus = ([], [], [])
        mut_i = int(round(ga_best_mut * 10000))
        for name, dist in instances.get(n, []):
            seed = make_seed('GA_SUM', n, 0, 0, ga_best_pop, mut_i)
            res = run_ga(dist, mut=ga_best_mut, pop=ga_best_pop, record_trace=False, seed=seed)
            scores.append(res['best_cost'])
            walls.append(res['wall'])
            cpus.append(res['cpu'])
        med_table[n]['GA'] = {'score': median_ignore_nan(scores), 'wall': median_ignore_nan(walls), 'cpu': median_ignore_nan(cpus)}
    for n in SIZES:
        scores, walls, cpus = ([], [], [])
        for name, dist in instances.get(n, []):
            res = run_astar(dist, time_limit_sec=TIME_LIMIT_ASTAR)
            scores.append(res['best_cost'])
            walls.append(res['wall'])
            cpus.append(res['cpu'])
        med_table[n]['A*'] = {'score': median_ignore_nan(scores), 'wall': median_ignore_nan(walls), 'cpu': median_ignore_nan(cpus)}
    plot(med_table)

def main():
    print('== Part III Experiments: HC / SA / GA + A* baseline ==')
    instances = instances_size()
    print('[INFO] HC sweep ...')
    hc_best_nr = hc_sweep(instances)
    print(f'[INFO] HC best restarts = {hc_best_nr}')
    print('[INFO] SA sweep ...')
    sa_best_alpha = sa_sweep(instances)
    print(f'[INFO] SA best alpha = {sa_best_alpha}')
    ga_fix_pop = GA_POP_GRID[len(GA_POP_GRID) // 2]
    print(f'[INFO] GA mutation sweep (fix pop={ga_fix_pop}) ...')
    ga_best_mut = ga_sweep_mut(instances, ga_fix_pop=ga_fix_pop)
    print(f'[INFO] GA best mutation = {ga_best_mut}')
    print(f'[INFO] GA population sweep (fix mut={ga_best_mut}) ...')
    ga_best_pop = ga_sweep_pop(instances, ga_fix_mut=ga_best_mut)
    print(f'[INFO] GA best population = {ga_best_pop}')
    representative_record(instances, hc_best_nr, sa_best_alpha, ga_best_mut, ga_best_pop)
    summary(instances, hc_best_nr, sa_best_alpha, ga_best_mut, ga_best_pop)
    print('[DONE] All outputs saved in ./outputs')
if __name__ == '__main__':
    main()