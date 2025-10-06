import random
import time
import sys

# --- 1. Problem Generation ---
def generate_3_sat(m, n):
    if not (isinstance(m, int) and isinstance(n, int)):
        print("Error: m and n must be integers.", file=sys.stderr)
        return []
    if m <= 0 or n <= 0:
        print("Error: m and n must be positive.", file=sys.stderr)
        return []
    if 3 > n:
        print("Error: k=3 cannot be greater than n.", file=sys.stderr)
        return []
    variables = list(range(1, n + 1))
    formula = []
    for _ in range(m):
        clause_vars = random.sample(variables, 3)
        clause = [var if random.choice([True, False]) else -var for var in clause_vars]
        formula.append(clause)
    return formula

# --- 2. Assignments and Heuristics ---
def generate_random_assignment(n):
    return [random.choice([True, False]) for _ in range(n)]

def evaluate_clause(clause, assignment):
    for literal in clause:
        var_index = abs(literal) - 1
        if (literal > 0 and assignment[var_index]) or (literal < 0 and not assignment[var_index]):
            return True
    return False

def h_satisfied_clauses(assignment, formula, weights=None):
    return sum(1 for clause in formula if evaluate_clause(clause, assignment))

def h_weighted_satisfied_clauses(assignment, formula, weights):
    score = 0
    for i, clause in enumerate(formula):
        if evaluate_clause(clause, assignment):
            score += weights[i]
    return score

# --- 3. Neighborhoods ---
def get_neighborhood_1_flip(assignment):
    neighbors = []
    for i in range(len(assignment)):
        neighbor = list(assignment)
        neighbor[i] = not neighbor[i]
        neighbors.append(neighbor)
    return neighbors

def get_neighborhood_k_flip(assignment, k):
    neighbors = []
    n = len(assignment)
    indices = list(range(n))
    for vars_to_flip in random.sample(list(set(tuple(sorted(combo)) for combo in [random.sample(indices, k) for _ in range(min(100, n**k))])), min(100, n**k)):
        neighbor = list(assignment)
        for idx in vars_to_flip:
            neighbor[idx] = not neighbor[idx]
        neighbors.append(neighbor)
    return neighbors

def find_best_neighbor(formula, neighborhood, heuristic_func, weights):
    best_neighbor = None
    best_score = -1
    evaluations = 0
    for neighbor in neighborhood:
        score = heuristic_func(neighbor, formula, weights)
        evaluations += 1
        if score > best_score:
            best_score = score
            best_neighbor = neighbor
    return best_neighbor, best_score, evaluations

# --- 4. Hill-Climbing ---
def hill_climbing(formula, n, heuristic_func, max_flips=1000, max_restarts=50):
    start_time = time.time()
    total_evaluations = 0
    for _ in range(max_restarts):
        current_assignment = generate_random_assignment(n)
        weights = [1] * len(formula)
        for flip_count in range(max_flips):
            current_score = heuristic_func(current_assignment, formula, weights)
            if current_score == len(formula):
                return current_assignment, flip_count, total_evaluations, time.time() - start_time
            neighborhood = get_neighborhood_1_flip(current_assignment)
            best_neighbor, best_score, evaluations = find_best_neighbor(formula, neighborhood, heuristic_func, weights)
            total_evaluations += evaluations
            if best_score <= current_score:
                break
            current_assignment = best_neighbor
            if heuristic_func == h_weighted_satisfied_clauses:
                for i, clause in enumerate(formula):
                    if not evaluate_clause(current_assignment, clause):
                        weights[i] += 1
    return None, max_flips * max_restarts, total_evaluations, time.time() - start_time

# --- 5. Beam Search ---
def beam_search(formula, n, heuristic_func, beam_width=3, max_flips=2000):
    start_time = time.time()
    total_evaluations = 0
    beam = [generate_random_assignment(n) for _ in range(beam_width)]
    weights = [1] * len(formula)
    for flip_count in range(max_flips):
        all_neighbors = []
        for assignment in beam:
            score = heuristic_func(assignment, formula, weights)
            if score == len(formula):
                return assignment, flip_count, total_evaluations, time.time() - start_time
            all_neighbors.extend(get_neighborhood_1_flip(assignment))
        if not all_neighbors:
            break
        neighbor_scores = [(heuristic_func(neighbor, formula, weights), neighbor) for neighbor in all_neighbors]
        total_evaluations += len(all_neighbors)
        unique_neighbors = sorted(list(set((s, tuple(n)) for s, n in neighbor_scores)), key=lambda x: x[0], reverse=True)
        beam = [list(n) for _, n in unique_neighbors[:beam_width]]
        if not beam:
            break
    return None, max_flips, total_evaluations, time.time() - start_time

# --- 6. Variable Neighborhood Descent (VND) ---
def variable_neighborhood_descent(formula, n, heuristic_func, max_flips=100):
    start_time = time.time()
    total_evaluations = 0
    current_assignment = generate_random_assignment(n)
    weights = [1] * len(formula)
    flip_count = 0
    while flip_count < max_flips:
        current_score = heuristic_func(current_assignment, formula, weights)
        if current_score == len(formula):
            return current_assignment, flip_count, total_evaluations, time.time() - start_time
        # VND: Try 1-flip, then 2-flip, then 3-flip neighborhoods
        improved = False
        for k in [1, 2, 3]:
            if k == 1:
                neighborhood = get_neighborhood_1_flip(current_assignment)
            else:
                # Sample up to 100 random k-flip neighbors
                neighborhood = []
                n_vars = len(current_assignment)
                for _ in range(min(100, n_vars**k)):
                    neighbor = list(current_assignment)
                    vars_to_flip = random.sample(range(n_vars), k)
                    for var_idx in vars_to_flip:
                        neighbor[var_idx] = not neighbor[var_idx]
                    neighborhood.append(neighbor)
            best_neighbor, best_score, evals = find_best_neighbor(formula, neighborhood, heuristic_func, weights)
            total_evaluations += evals
            if best_score > current_score:
                current_assignment = best_neighbor
                flip_count += 1
                improved = True
                break  # Go back to k=1
        if not improved:
            # Restart with new assignment if stuck
            current_assignment = generate_random_assignment(n)
            weights = [1] * len(formula)
            flip_count += 1
    return None, flip_count, total_evaluations, time.time() - start_time

# --- 7. Experiment Runner ---
def run_experiments():
    N_VALUES = [20, 40]
    M_N_RATIOS = [3.8, 4.26, 4.6]
    NUM_INSTANCES = 5
    heuristics = {
        "H1 (Satisfied)": h_satisfied_clauses,
        "H2 (Weighted)": h_weighted_satisfied_clauses,
    }
    random.seed(time.time())
    print("="*80)
    print("3-SAT Local Search Algorithm Comparison")
    print("="*80)
    for n in N_VALUES:
        for ratio in M_N_RATIOS:
            m = int(n * ratio)
            print(f"\n--- n={n}, m={m} (ratio={ratio:.2f}) ---\n")
            solvers = {
                "Hill-Climbing": (hill_climbing, {}),
                "Beam-Search (w=3)": (beam_search, {'beam_width': 3}),
                "Beam-Search (w=4)": (beam_search, {'beam_width': 4}),
                "VND": (variable_neighborhood_descent, {}),
            }
            for h_name, h_func in heuristics.items():
                print(f"  Heuristic: {h_name}")
                for s_name, (s_func, s_args) in solvers.items():
                    results = {'success': 0, 'flips': [], 'time': [], 'penetrance': []}
                    for _ in range(NUM_INSTANCES):
                        formula = generate_3_sat(m, n)
                        solution, flips, evals, duration = s_func(formula, n, h_func, **s_args)
                        if solution:
                            results['success'] += 1
                            results['flips'].append(flips)
                            results['time'].append(duration)
                            if evals > 0:
                                pen = flips / evals if flips > 0 else 0
                                results['penetrance'].append(pen)
                    success_rate = (results['success'] / NUM_INSTANCES) * 100
                    avg_flips = sum(results['flips']) / len(results['flips']) if results['flips'] else 0
                    avg_time = sum(results['time']) / len(results['time']) if results['time'] else 0
                    avg_pen = sum(results['penetrance']) / len(results['penetrance']) if results['penetrance'] else 0
                    print(f"    {s_name:<20} | Success: {success_rate:6.1f}% | "
                          f"Avg Flips: {avg_flips:8.1f} | "
                          f"Avg Time: {avg_time:7.4f}s | "
                          f"Avg Penetrance: {avg_pen:.5f}")
            print("-" * 80)

if __name__ == '__main__':
    run_experiments()
