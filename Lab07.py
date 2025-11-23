from typing import List, Tuple, Dict, Optional
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


#----- Utilities for Tic-Tac-Toe------------

WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
             (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]


def winner(board: List[str]) -> Optional[str]:
    
    for a, b, c in WIN_LINES:
        if board[a] != ' ' and board[a] == board[b] == board[c]:
            return board[a]
    if ' ' not in board:
        return 'D'
    return None


def transforms(board: Tuple[str, ...]) -> List[Tuple[str, ...]]:
   
    
    def to_grid(b): return [list(b[0:3]), list(b[3:6]), list(b[6:9])]
    def from_grid(g): return tuple(sum(g, []))

    grid = to_grid(board)
    variants = []
    current = grid
    for r in range(4):
        if r > 0:
            # rotate 90 degrees clockwise
            current = [list(row) for row in zip(*current[::-1])]
        variants.append(from_grid(current))
      
        variants.append(from_grid([row[::-1] for row in current]))

    unique = []
    for v in variants:
        if v not in unique:
            unique.append(v)
    return unique


def canonical(board: Tuple[str, ...]) -> Tuple[str, ...]:
    
    return min(transforms(board))


#-------- MENACE agent-----------

class MENACE:
   

    def _init_(self,
                 initial_beads: int = 3,
                 reward_win: int = 3,
                 reward_draw: int = 1,
                 punish_loss: int = 1):
      
        self.boxes: Dict[Tuple[str, ...], Dict[int, int]] = {}
        self.initial_beads = initial_beads
        self.reward_win = reward_win
        self.reward_draw = reward_draw
        self.punish_loss = punish_loss
        self.history: List[Tuple[Tuple[str, ...], int]] = [] 

    def ensure_box(self, board: Tuple[str, ...]) -> Tuple[str, ...]:
       
        key = canonical(board)
        if key not in self.boxes:
            legal_moves = [i for i, v in enumerate(key) if v == ' ']
            # start each legal action with initial bead count
            self.boxes[key] = {a: int(self.initial_beads) for a in legal_moves}
        return key

    def select_move(self, board: Tuple[str, ...]) -> int:
        
        key = self.ensure_box(board)
        action_weights = self.boxes[key]
        actions = list(action_weights.keys())
        weights = [action_weights[a] for a in actions]

        # If beads accidentally sum to 0 (shouldn't normally happen), reset to safe positive
        if sum(weights) <= 0:
            for a in actions:
                action_weights[a] = max(1, int(self.initial_beads))
            weights = [action_weights[a] for a in actions]

        choice = random.choices(actions, weights=weights, k=1)[0]
        # remember which box/action was used in this game so we can update later
        self.history.append((key, choice))
        return choice

    def update(self, result: str) -> None:
               if result == 'X':
            delta = self.reward_win
        elif result == 'D':
            delta = self.reward_draw
        else:
            delta = -self.punish_loss

        for key, action in self.history:
            if key in self.boxes and action in self.boxes[key]:
                new_count = self.boxes[key][action] + delta
                # keep counts non-negative
                self.boxes[key][action] = max(0, int(new_count))
        # clear history for next game
        self.history.clear()


def play_menace_game(agent: MENACE, verbose: bool = False) -> str:
       board = [' '] * 9
    turn = 'X'
    while True:
        if turn == 'X':
            # MENACE expects a tuple board as key
            move = agent.select_move(tuple(board))
        else:
            legal = [i for i, v in enumerate(board) if v == ' ']
            move = random.choice(legal)

        board[move] = turn
        if verbose:
            print(f"{turn} -> {move}  | {'|'.join(board[0:3])} ...")

        w = winner(board)
        if w is not None:
            agent.update(w if turn in ('X', 'O') or w == 'D' else w)
            return w
        turn = 'O' if turn == 'X' else 'X'


# -----------------------
# Stationary 2-armed bandit (epsilon-greedy)
# -----------------------
def run_binary_bandit(p1: float,
                      p2: float,
                      eps: float = 0.1,
                      steps: int = 5000,
                      seed: Optional[int] = None):
        if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    Q = [0.0, 0.0]   # estimated action values
    N = [0, 0]       # counts of selections
    rewards = []

    for t in range(1, steps + 1):
        # choose: explore with probability eps; otherwise exploit
        if random.random() < eps:
            action = random.choice([0, 1])
        else:
            action = 0 if Q[0] >= Q[1] else 1

        # sample reward from Bernoulli with p1 or p2
        prob = p1 if action == 0 else p2
        reward = 1 if random.random() < prob else 0

        # incremental sample-average update
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

        rewards.append(reward)

    avg_reward = float(np.mean(rewards))
    return Q, N, avg_reward, rewards


# -------------Nonstationary K-armed bandit -------------
class NonStationaryBandit:
    

    def _init_(self,
                 k: int = 10,
                 mu0: float = 0.0,
                 walk_std: float = 0.01,
                 reward_std: float = 1.0,
                 seed: Optional[int] = None):
        self.k = k
        self.mu = np.zeros(k) + float(mu0)
        self.walk_std = float(walk_std)
        self.reward_std = float(reward_std)
        self.rng = np.random.RandomState(seed)
        self.time = 0

    def step(self, action: int) -> float:
        
        reward = float(self.rng.normal(self.mu[action], self.reward_std))
        # random walk update to all means
        self.mu += self.rng.normal(0.0, self.walk_std, size=self.k)
        self.time += 1
        return reward

    def get_true_means(self) -> np.ndarray:
        return self.mu.copy()



#-------- Nonstationary agent runner

def run_agent_nonstationary(agent_type: str = 'sample_avg',
                            eps: float = 0.1,
                            alpha: float = 0.1,
                            steps: int = 10000,
                            trials: int = 30,
                            walk_std: float = 0.01,
                            seed_base: int = 0):
    
    
    k = 10
    rewards_all = np.zeros((trials, steps))
    optimal_all = np.zeros((trials, steps))

    for trial in range(trials):
        bandit = NonStationaryBandit(k=k, walk_std=walk_std, seed=seed_base + trial)
        Q = np.zeros(k)
        N = np.zeros(k, dtype=int)

        for t in range(steps):
            # action selection (epsilon-greedy)
            if random.random() < eps:
                action = random.randint(0, k - 1)
            else:
                action = int(np.argmax(Q))

            reward = bandit.step(action)

            # update rule
            if agent_type == 'sample_avg':
                N[action] += 1
                Q[action] += (reward - Q[action]) / N[action]
            else:  # constant-alpha update
                Q[action] += alpha * (reward - Q[action])

            rewards_all[trial, t] = reward
            true_means = bandit.get_true_means()
            optimal_action = int(np.argmax(true_means))
            optimal_all[trial, t] = 1 if action == optimal_action else 0

    avg_rewards = rewards_all.mean(axis=0)
    avg_optimal = optimal_all.mean(axis=0)
    return avg_rewards, avg_optimal


def smooth(x: np.ndarray, window: int = 50) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def main():
    # reproducibility
    random.seed(0)
    np.random.seed(0)

    # ----- 1) Train MENACE -----
    print(" — Training MENACE vs random opponent")
    print("Training for 10,000 games, reporting in batches of 1,000...")
    menace_agent = MENACE(initial_beads=3, reward_win=3, reward_draw=1, punish_loss=1)

    wins = draws = losses = 0
    batch_stats = []
    total_games = 10_000
    batch_size = 1_000
    for i in range(total_games):
        result = play_menace_game(menace_agent)
        if result == 'X':
            wins += 1
        elif result == 'O':
            losses += 1
        else:
            draws += 1

        if (i + 1) % batch_size == 0:
            batch_stats.append((i + 1, wins, draws, losses))

    print("MENACE training batches (games_played, wins, draws, losses):")
    for stats in batch_stats:
        print(stats)

    # -----  Stationary binary bandit experiments -----
    print("\n — Stationary 2-armed bandit (Bernoulli arms p1=0.6, p2=0.4)")
    for eps in [0.0, 0.01, 0.1, 0.2]:
        Q, N, avg_r, _ = run_binary_bandit(0.6, 0.4, eps=eps, steps=5000, seed=42)
        print(f"eps={eps:0.2f} -> Q={Q}, counts={N}, avg_reward={avg_r:.4f}")

    # ----- Nonstationary experiments -----
    print("\n — Nonstationary 10-armed bandit comparison")
    steps = 10_000
    trials = 30
    eps = 0.1
    alpha = 0.1
    walk_std = 0.01

    print("Running sample-average agent...")
    rewards_sa, opt_sa = run_agent_nonstationary('sample_avg', eps=eps, alpha=alpha,
                                                 steps=steps, trials=trials, walk_std=walk_std,
                                                 seed_base=0)
    print("Running constant-alpha agent...")
    rewards_ca, opt_ca = run_agent_nonstationary('const_alpha', eps=eps, alpha=alpha,
                                                 steps=steps, trials=trials, walk_std=walk_std,
                                                 seed_base=1000)

   

if __name__ == "__main__":
    main()
