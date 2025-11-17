#python3 -m venv .venv
#source .venv/bin/activate
#pip install -U pip
#pip install numpy  matplotlib
#python3 main.py
#or
#python3 main.py --file ./mats_911/n_random_adj_mat_i.txt

import time
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_DIR = Path("./mats_911")
OUTPUT_DIR = Path("./outputs")
SIZES = [5, 10, 15, 20, 25, 30]
K_GRID = [1, 2, 3, 5, 8, 10, 12, 15]
REPEATS_GRID = [1, 2, 5, 10, 15, 20, 30, 50, 100]


RECORD = {
    "k_sweep": [],
    "num_repeats_sweep": [],
    "compare": []
}

def read_matrix(input, sizes):
    mats = {n: [] for n in sizes}
    for p in sorted(input.glob("*.txt")):
        mat = np.loadtxt(p, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            continue
        n = mat.shape[0]
        if n not in mats:
            continue
        if np.allclose(mat, mat.T):
            mat = 0.5 * (mat + mat.T)
        else:
            if np.allclose(np.tril(mat, -1), 0.0) or np.allclose(np.triu(mat, 1), 0.0):
                fullmat = mat+mat.T - np.diag(np.diag(mat))
                mat = fullmat
        np.fill_diagonal(mat, 0.0)
        mats[n].append(mat)

    return mats
    

def travel_length(mat: np.ndarray, travel_route):
    return float(sum(mat[travel_route[i], travel_route[i+1]] for i in range(len(travel_route )-1)))


def nn(mat: np.ndarray, start = 0):
    n = mat.shape[0]
    visited = np.zeros(n, dtype= bool)
    travel_route = [start]
    visited[start] = True
    cur = start
    
    for _ in range(n - 1):
        visited[cur] = True
        distance = np.where(~visited, mat[cur], np.inf)
        next_city = int(np.argmin(distance))
        travel_route.append(next_city)
        visited[next_city] = True
        cur = next_city

    travel_route.append(start)
    return travel_route

def two_opt(mat: np.ndarray, travel_route):
    n = len(travel_route) - 1 #excluding start point
    improved = True
    while improved:
        improved = False
        for i in range (1, n - 1):
            for j in range(i + 1, n):
                a, b = travel_route[i-1], travel_route[i]
                x, y = travel_route[j], travel_route[(j+1)%(n+1)]
                delta = (mat[a, x] + mat[b, y]) - (mat[a, b] + mat[x, y])
                if delta < 0:
                    travel_route[i: j + 1] = reversed(travel_route[i : j+1])
                    improved = True
                    break

    return travel_route

def nn_two_opt(mat: np.ndarray, start=0):
    travel_route = nn(mat, start)
    return two_opt(mat, travel_route)

def rrnn(mat:np.ndarray, k=5, num_repeats=50, seed = 12345):
    n = mat.shape[0]
    rng = np.random.default_rng(seed)
    best_route, best_cost = None, np.inf
    for _ in range(num_repeats):
        start = int(rng.integers(0, n))
        visited = np.zeros(n, dtype = bool)
        travel_route = [start]
        visited[start] = True
        cur = start
        for steps in range(n-1):
            distance = np.where(~visited, mat[cur], np.inf)
            distance[cur] = np.inf
            candidate = np.where(np.isfinite(distance))[0]
            num_candidate = min(k, candidate.size)
            part = np.argpartition(distance[candidate], num_candidate-1)[:num_candidate]
            k_candidate = candidate[part]
            next_city = int(rng.choice(k_candidate))
            travel_route.append(next_city)
            visited[next_city ] = True
            cur = next_city

        travel_route.append(start)
        travel_route = two_opt(mat, travel_route)

        cost = float(sum(mat[travel_route[i], travel_route[i+1]] for i in range(len(travel_route)-1)))
        if cost<best_cost:
            best_cost = cost
            best_route = travel_route
    return best_route

def run_record(mat: np.ndarray, algo, *, start = 0, k = 5, num_repeats = 10, seed = 12345):
    def run():
        if algo == "nn":
            travel_route = nn(mat, start = start)
        elif algo == "nn2opt":
            travel_route = nn_two_opt(mat, start=start)
        else:
            travel_route = rrnn(mat, k=k, num_repeats=num_repeats, seed=seed)

        return travel_route

    walltime0=time.time_ns()
    cputime0 = time.process_time_ns()

    travel_route = run()
    
    walltime1 = time.time_ns()
    cputime1 = time.process_time_ns()

    wallclocktime = (walltime1 - walltime0) / 1e9
    cpuclocktime = (cputime1 - cputime0) / 1e9

    if cpuclocktime == 0.0:
        for reps in (10, 100, 1000):
            wall0 = time.time_ns()
            cpu0 = time.process_time_ns()
            for i in range(reps):
                travel_route = run()
            wall1 = time.time_ns()
            cpu1 = time.process_time_ns()

            wall = (wall1 - wall0) / 1e9 / reps
            cpu = (cpu1 - cpu0) / 1e9 / reps
            if cpu>0.0:
                wallclocktime = wall
                cpuclocktime = cpu
                break

    return travel_length(mat, travel_route), wallclocktime, cpuclocktime

def median(values):
    return float(np.median(np.asarray(values, dtype=float)))

def make_csv(path: Path, rows, header):
    with open(path, "w", newline="") as f:
        w= csv.writer(f)
        w.writerow(header)
        w.writerows(rows)




def rrnn_k(mats):
    scores = []
    for k in K_GRID:
        costs = []
        for n in SIZES:
            for mat in mats[n]:
                cost, wall, cpu = run_record(mat, "rrnn",k=k, num_repeats = 10, seed = 12345)
                RECORD["k_sweep"].append(("rrnn", k, n, cost, wall, cpu))
                costs.append(cost)
        scores.append(median(costs))
    fig = plt.figure()
    plt.plot(K_GRID, scores, marker="o")
    plt.xlabel("k - RRNN")
    plt.ylabel("Median Cost")
    plt.title("RRNN k-sweep")
    fig.savefig(OUTPUT_DIR / "rrnn_k_sweep.png", dpi = 160, bbox_inches=  "tight")
    plt.close(fig)

    best_k = K_GRID[int(np.argmin(np.asarray(scores)))]
    return best_k

def rrnn_num_repeats(mats, k):
    scores = []
    for r in REPEATS_GRID:
        costs = []
        for n in SIZES:
            for mat in mats[n]:
                cost, wall, cpu = run_record(mat, "rrnn", k = k, num_repeats=r, seed = 12345)
                RECORD["num_repeats_sweep"].append(("rrnn", k, r, n, cost, wall, cpu))
                costs.append(cost)
        scores.append(median(costs))
    fig = plt.figure()
    plt.plot(REPEATS_GRID, scores, marker = "o")
    plt.xlabel("num_repeats") 
    plt.ylabel("Median Cost")
    plt.title("RRNN num_repeats-sweep")
    fig.savefig(OUTPUT_DIR / "rrnn_num_repeats", dpi=160, bbox_inches = "tight" )
    plt.close(fig)

    best_num_repeats = REPEATS_GRID[int(np.argmin(np.asarray(scores)))]
    return best_num_repeats

def compare(mats, best_k, best_repeats):
    median_cost = {"nn": [], "nn2opt": [], "rrnn": []}
    median_wall_time = {"nn": [], "nn2opt": [], "rrnn": []}
    median_cpu_time = {"nn": [], "nn2opt": [], "rrnn": []}

    for n in SIZES:
        walltimes_nn, cputimes_nn, costs_nn = [], [] ,[]
        walltimes_nn_two_opt, cputimes_nn_two_opt, costs_nn_two_opt = [], [], []
        walltimes_rrnn, cputimes_rrnn, costs_rrnn = [], [],[]
        for mat in mats[n]:
            cost, wall, cpu = run_record(mat, "nn", k=best_k, num_repeats=best_repeats, seed=12345)
            RECORD["compare"].append(("nn", n, cost, wall, cpu))
            costs_nn.append(cost)
            walltimes_nn.append(wall)
            cputimes_nn.append(cpu)

            cost, wall, cpu = run_record(mat, "nn2opt", k=best_k, num_repeats=best_repeats, seed=12345)
            RECORD["compare"].append(("nn2opt", n, cost, wall, cpu))
            costs_nn_two_opt.append(cost)
            walltimes_nn_two_opt.append(wall)
            cputimes_nn_two_opt.append(cpu)

            cost, wall, cpu = run_record(mat, "rrnn", k=best_k, num_repeats=best_repeats, seed = 12345)
            RECORD["compare"].append(("rrnn", n, cost, wall, cpu))
            costs_rrnn.append(cost)
            walltimes_rrnn.append(wall)
            cputimes_rrnn.append(cpu)

        median_cost["nn"].append(median(costs_nn))
        median_wall_time["nn"].append(median(walltimes_nn))
        median_cpu_time["nn"].append(median(cputimes_nn))
        median_cost["nn2opt"].append(median(costs_nn_two_opt))
        median_wall_time["nn2opt"].append(median(walltimes_nn_two_opt))
        median_cpu_time["nn2opt"].append(median(cputimes_nn_two_opt))
        median_cost["rrnn"].append(median(costs_rrnn))
        median_wall_time["rrnn"].append(median(walltimes_rrnn))
        median_cpu_time["rrnn"].append(median(cputimes_rrnn))


    fig_cost = plt.figure()
    for algo, label in [("nn","NN"), ("nn2opt","NN+2Opt"), ("rrnn","RRNN")]:
        plt.plot(SIZES, median_cost[algo], marker="o", label=label)
    plt.xlabel("Problem size (n)"); plt.ylabel("Median cost"); plt.title("Median Cost vs n")
    plt.legend()
    fig_cost.savefig(OUTPUT_DIR / "compare_cost.png", dpi=160, bbox_inches="tight"); plt.close(fig_cost)

    fig_wall = plt.figure()
    for algo, label in [("nn","NN"), ("nn2opt","NN+2Opt"), ("rrnn","RRNN")]:
        plt.plot(SIZES, median_wall_time[algo], marker="o", label=label)
    plt.xlabel("Problem size (n)"); plt.ylabel("Median wall time (s)"); plt.title("Median Wall Time vs n")
    plt.legend()
    fig_wall.savefig(OUTPUT_DIR / "compare_wall.png", dpi=160, bbox_inches="tight"); plt.close(fig_wall)

    fig_cpu = plt.figure()
    for algo, label in [("nn","NN"), ("nn2opt","NN+2Opt"), ("rrnn","RRNN")]:
        plt.plot(SIZES, median_cpu_time[algo], marker="o", label=label)
    plt.xlabel("Problem size (n)"); plt.ylabel("Median CPU time (s)"); plt.title("Median CPU Time vs n")
    plt.legend()
    fig_cpu.savefig(OUTPUT_DIR / "compare_cpu.png", dpi=160, bbox_inches="tight"); plt.close(fig_cpu)


def run_single_file(file_path):
    import shutil
    global SIZES, OUTPUT_DIR, RECORD


    p = Path(file_path)
    mat = np.loadtxt(p, dtype=float)
    n = mat.shape[0]

    temp_dir = OUTPUT_DIR / f"single_temp_{p.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, temp_dir / p.name)
    
    mat = read_matrix(temp_dir, [n])

    old_sizes = SIZES[:]
    old_output = OUTPUT_DIR
    old_record = {k: list(v) for k, v in RECORD.items()}

    single_out = OUTPUT_DIR / f"single_{p.stem}"
    single_out.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = single_out
    SIZES = [n ]
    for k in RECORD:
        RECORD[k].clear()
    try:
        print("RRNN k-sweep")
        best_k = rrnn_k(mat)
        print(f"best_k = {best_k}")

        print("RRNN num_repeats-sweep")
        best_repeats = rrnn_num_repeats(mat, best_k)
        print(f"best_num_repeats = {best_repeats}")

        print("Comapre NN / NN2Opt / RRNN")
        compare(mat, best_k, best_repeats)

        make_csv(OUTPUT_DIR / "k_sweep.csv",
                 RECORD["k_sweep"],
                 ["algo", "k", "n", "cost", "wall", "cpu"])
        make_csv(OUTPUT_DIR / "num_repeats_sweep.csv",
                 RECORD["num_repeats_sweep"],
                 ["algo", "k", "n", "cost", "wall", "cpu"])
        make_csv(OUTPUT_DIR / "compare.csv",
                 RECORD["compare"],
                 ["algo", "k", "n", "cost", "wall", "cpu"])
        
        print(f"[single-file] Figures and CSVs saved")


    finally:
        OUTPUT_DIR = old_output
        SIZES = old_sizes
        RECORD = old_record
        try: 
            shutil.rmtree(temp_dir)
        except Exception:
            pass


                



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mats = read_matrix(INPUT_DIR, SIZES)

    print("RRNN k-sweep")
    best_k = rrnn_k(mats)
    print(f"best_k = {best_k}")

    print("RRNN num_repeats-sweep")
    best_repeats = rrnn_num_repeats(mats, best_k)
    print(f"best_num_repeats = {best_repeats}")
    
    print("Compare NN / NN+2opt / RRNN")
    compare(mats, best_k, best_repeats)



    make_csv(OUTPUT_DIR / "k_sweep.csv",
             RECORD["k_sweep"],
             ["algo", "k", "n", "cost", "wall", "cpu"])
    
    make_csv(OUTPUT_DIR / "num_repeats_sweep.csv",
             RECORD["num_repeats_sweep"],
             ["algo", "k", "n", "cost", "wall", "cpu"])
    make_csv(OUTPUT_DIR / "compare.csv",
             RECORD["compare"],
             ["algo", "k", "n", "cost", "wall", "cpu"])

    print("Figures and CSVs saved") 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description= "Part1 Single file load")
    parser.add_argument("--file", type=str, help="Single file path (ex: ./mats_911/n_random_adj_mat_01.txt)")
    args = parser.parse_args()

    if args.file:
        run_single_file(args.file)
    else:
        main()