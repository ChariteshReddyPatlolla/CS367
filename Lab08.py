from typing import Dict, Tuple, List
import argparse
import math
from math import factorial
import numpy as np


#------ — GRIDWORLD VALUE ITERATION-------------

ROWS, COLS = 3, 4
TERMINAL_STATES: Dict[Tuple[int, int], float] = {(0, 3): 1.0, (1, 3): -1.0}
WALL = (1, 1)

# Actions are labeled and give (dr, dc)
ACTIONS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

# Dynamics probabilities 
INTENDED = 0.8
LEFT_NOISE = 0.1
RIGHT_NOISE = 0.1

DISCOUNT_GW = 0.99
VALUE_ITER_ITERS = 200


def in_bounds(r: int, c: int) -> bool:
    """Return True if position (r,c) is a valid non-wall grid cell."""
    return 0 <= r < ROWS and 0 <= c < COLS and (r, c) != WALL


def action_outcomes(r: int, c: int, action: str):
    """
    Given a cell (r,c) and an intended action label (U/D/L/R),
    return a list of (probability, (nr,nc)) outcomes accounting for noise.
    """
    moves = {
        "U": [("U", INTENDED), ("L", LEFT_NOISE), ("R", RIGHT_NOISE)],
        "D": [("D", INTENDED), ("R", LEFT_NOISE), ("L", RIGHT_NOISE)],
        "L": [("L", INTENDED), ("D", LEFT_NOISE), ("U", RIGHT_NOISE)],
        "R": [("R", INTENDED), ("U", LEFT_NOISE), ("D", RIGHT_NOISE)]
    }

    outcomes = []
    for move_label, p in moves[action]:
        dr, dc = ACTIONS[move_label]
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc):
            # If action would hit wall/boundary, agent stays in place
            nr, nc = r, c
        outcomes.append((p, (nr, nc)))
    return outcomes


def value_iteration(reward_step: float,
                    discount: float = DISCOUNT_GW,
                    iters: int = VALUE_ITER_ITERS) -> np.ndarray:
    
    V = np.zeros((ROWS, COLS))
    # initialize terminal values explicitly
    for pos, val in TERMINAL_STATES.items():
        V[pos] = val

    for it in range(iters):
        newV = V.copy()
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in terminal_state or (r, c) == wall:
                    continue  # skip terminal & wall
                action_values = []
                for a in ACTIONS:
                    val = 0.0
                    for p, (nr, nc) in action_outcomes(r, c, a):
                        
                        step_reward = reward_step if (nr, nc) not in terminal_states else terminal_states[(nr, nc)]
                        val += p * (step_reward + discount * V[nr, nc])
                    action_values.append(val)
                newV[r, c] = max(action_values)
        V = newV
    return V



#  —------------- G-BIKE RENTAL--------------

# Problem constants 
MAX_BIKES = 20
MOVE_LIMIT = 5        # max bikes moved overnight
RENT_REWARD = 10      # reward per rental
MOVE_COST = 2         # per-bike moving cost
DISCOUNT_GBIKE = 0.9

# Poisson rental
RENT1_LAMBDA = 3.0
RENT2_LAMBDA = 4.0
RET1_LAMBDA = 3.0
RET2_LAMBDA = 2.0

_poisson_cache: Dict[Tuple[int, float], float] = {}


def poisson_pmf(n: int, lam: float) -> float:
   
    key = (n, lam)
    if key not in _poisson_cache:
        _poisson_cache[key] = math.exp(-lam) * lam**n / factorial(n)
    return _poisson_cache[key]


def expected_return_gbike(state: Tuple[int, int],
                          action: int,
                          V: np.ndarray,
                          max_rent_consider: int = 10) -> float:
   
    bikes1, bikes2 = state
    # apply action (move bikes overnight)
    bikes1 = min(MAX_BIKES, max(0, bikes1 - action))
    bikes2 = min(MAX_BIKES, max(0, bikes2 + action))

    
    reward = -MOVE_COST * abs(action)
    expected_future_value = 0.0
    expected_reward_rentals = 0.0

    # sum over plausible rental and return counts 
    for r1 in range(0, max_rent_consider + 1):
        p_r1 = poisson_pmf(r1, RENT1_LAMBDA)
        for r2 in range(0, max_rent_consider + 1):
            p_r2 = poisson_pmf(r2, RENT2_LAMBDA)
            for ret1 in range(0, max_rent_consider + 1):
                p_ret1 = poisson_pmf(ret1, RET1_LAMBDA)
                for ret2 in range(0, max_rent_consider + 1):
                    p_ret2 = poisson_pmf(ret2, RET2_LAMBDA)
                    p = p_r1 * p_r2 * p_ret1 * p_ret2
                    if p <= 1e-8:
                        continue  

                    real_r1 = min(bikes1, r1)
                    real_r2 = min(bikes2, r2)
                    reward_r = (real_r1 + real_r2) * RENT_REWARD

                    next1 = min(MAX_BIKES, bikes1 - real_r1 + ret1)
                    next2 = min(MAX_BIKES, bikes2 - real_r2 + ret2)

                    expected_reward_rentals += p * reward_r
                    expected_future_value += p * V[next1, next2]

    return reward + expected_reward_rentals + DISCOUNT_GBIKE * expected_future_value


def policy_iteration_gbike(max_rent_consider: int = 10,
                           eval_iters: int = 20,
    
    V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    policy = np.zeros_like(V, dtype=int)
    stable = False
    iteration = 0

    while not stable and iteration < max_policy_iters:
        iteration += 1
        # Policy evaluation (truncated number of sweeps)
        for _ in range(eval_iters):
            newV = np.zeros_like(V)
            for i in range(MAX_BIKES + 1):
                for j in range(MAX_BIKES + 1):
                    action = int(policy[i, j])
                    newV[i, j] = expected_return_gbike((i, j), action, V, max_rent_consider=max_rent_consider)
            V = newV

        
        stable = True
        changed = 0
        for i in range(MAX_BIKES + 1):
            for j in range(MAX_BIKES + 1):
                # consider allowable actions given constraints
                q_values = []
                for a in range(-MOVE_LIMIT, MOVE_LIMIT + 1):
                    if 0 <= i - a <= MAX_BIKES and 0 <= j + a <= MAX_BIKES:
                        q = expected_return_gbike((i, j), a, V, max_rent_consider=max_rent_consider)
                        q_values.append((q, a))
                # pick best action
                best_q, best_a = max(q_values)
                if best_a != policy[i, j]:
                    policy[i, j] = best_a
                    stable = False
                    changed += 1
        print(f"Policy iteration #{iteration} finished; actions changed: {changed}")
        if changed == 0:
            stable = True

    return policy, V

#-----------Modified G-Bike----------

def expected_return_gbike_modified(state: Tuple[int, int],
                                   action: int,
                                   V: np.ndarray,
   
    bikes1, bikes2 = state
    # free shuttle: first bike moved costs nothing, only extra moves cost
    free_move_cost = max(0, abs(action) - 1) * MOVE_COST

    
    bikes1 = min(MAX_BIKES, max(0, bikes1 - action))
    bikes2 = min(MAX_BIKES, max(0, bikes2 + action))

    
    parking_cost = 0.0
    if bikes1 > 10:
        parking_cost -= 4.0
    if bikes2 > 10:
        parking_cost -= 4.0

    reward = parking_cost - free_move_cost
    expected_future_value = 0.0
    expected_reward_rentals = 0.0

    for r1 in range(0, max_rent_consider + 1):
        p_r1 = poisson_pmf(r1, RENT1_LAMBDA)
        for r2 in range(0, max_rent_consider + 1):
            p_r2 = poisson_pmf(r2, RENT2_LAMBDA)
            for ret1 in range(0, max_rent_consider + 1):
                p_ret1 = poisson_pmf(ret1, RET1_LAMBDA)
                for ret2 in range(0, max_rent_consider + 1):
                    p_ret2 = poisson_pmf(ret2, RET2_LAMBDA)
                    p = p_r1 * p_r2 * p_ret1 * p_ret2
                    if p <= 1e-8:
                        continue

                    real_r1 = min(bikes1, r1)
                    real_r2 = min(bikes2, r2)
                    reward_r = (real_r1 + real_r2) * RENT_REWARD

                    next1 = min(MAX_BIKES, bikes1 - real_r1 + ret1)
                    next2 = min(MAX_BIKES, bikes2 - real_r2 + ret2)

                    expected_reward_rentals += p * reward_r
                    expected_future_value += p * V[next1, next2]

    return reward + expected_reward_rentals + DISCOUNT_GBIKE * expected_future_value


def policy_iteration_gbike_modified(max_rent_consider: int = 10,
                                    eval_iters: int = 20,
   
    V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    policy = np.zeros_like(V, dtype=int)
    stable = False
    iteration = 0

    while not stable and iteration < max_policy_iters:
        iteration += 1
        # policy evaluation
        for _ in range(eval_iters):
            newV = np.zeros_like(V)
            for i in range(MAX_BIKES + 1):
                for j in range(MAX_BIKES + 1):
                    action = int(policy[i, j])
                    newV[i, j] = expected_return_gbike_modified((i, j), action, V, max_rent_consider=max_rent_consider)
            V = newV

        # policy improvement
        stable = True
        changed = 0
        for i in range(MAX_BIKES + 1):
            for j in range(MAX_BIKES + 1):
                q_values = []
                for a in range(-MOVE_LIMIT, MOVE_LIMIT + 1):
                    if 0 <= i - a <= MAX_BIKES and 0 <= j + a <= MAX_BIKES:
                        q = expected_return_gbike_modified((i, j), a, V, max_rent_consider=max_rent_consider)
                        q_values.append((q, a))
                best_q, best_a = max(q_values)
                if best_a != policy[i, j]:
                    policy[i, j] = best_a
                    stable = False
                    changed += 1
        print(f"Modified policy iteration #{iteration} finished; actions changed: {changed}")
        if changed == 0:
            stable = True

    return policy, V


def run_gridworld_demo():
    print("=== GRIDWORLD VALUE ITERATION ===")
    sample_rewards = [-2.0, 0.1, 0.02, 1.0]
    results = {}
    for r in sample_rewards:
        print(f"\nComputing V for step reward r = {r}")
        V = value_iteration(reward_step=r)
        results[r] = V
        print(np.round(V, 3))
    print("\nGridworld runs complete.")


def run_gbike_demo(run_original: bool = False, run_modified: bool = False):
    if run_original:
        print("\n=== RUN ORIGINAL G-BIKE POLICY ITERATION ===")
        print("This is slow — expect several minutes depending on machine & parameters.")
        policy, V = policy_iteration_gbike(max_rent_consider=8, eval_iters=10, max_policy_iters=30)
        print("Original G-Bike policy iteration finished.")
        
    if run_modified:
        print("\n=== RUN MODIFIED G-BIKE POLICY ITERATION ===")
        print("This is slow — expect several minutes depending on machine & parameters.")
        policy2, V2 = policy_iteration_gbike_modified(max_rent_consider=8, eval_iters=10, max_policy_iters=30)
        print("Modified G-Bike policy iteration finished.")
        


def parse_cli_and_run():
    parser = argparse.ArgumentParser(description="Friendly RL exercises runner")
    parser.add_argument("--grid", action="store_true", help="Run Gridworld value iteration demos")
    parser.add_argument("--gbike", action="store_true", help="Run original G-Bike policy iteration (slow)")
    parser.add_argument("--gbike-mod", action="store_true", help="Run modified G-Bike policy iteration (slow)")
    parser.add_argument("--all", action="store_true", help="Run all sections (grid + both gbike variants)")
    args = parser.parse_args()

    if args.all:
        run_gridworld_demo()
        run_gbike_demo(run_original=True, run_modified=True)
    else:
        if args.grid:
            run_gridworld_demo()
        if args.gbike or args.gbike_mod:
            run_gbike_demo(run_original=args.gbike, run_modified=args.gbike_mod)

    if not (args.grid or args.gbike or args.gbike_mod or args.all):
        print("No flags supplied. Try --grid or --gbike or --gbike-mod. Use -h for help.")


if __name__ == "__main__":
    parse_cli_and_run()
