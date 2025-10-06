import os
import time
import psutil
import copy
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque
from tqdm import tqdm

def load_mat_image(file_path):
    matrix_data = []
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()[5:]
        for line in lines:
            line = line.strip()
            if line:
                matrix_data.append(int(line))
    except FileNotFoundError:
        return None
    matrix = np.array(matrix_data)
    if matrix.size != 512 * 512:
        return None
    return matrix.reshape((512, 512)).T

def create_patches(image, patch_size=128):
    tile_map = {}
    num_tiles = image.shape[0] // patch_size
    tile_id = 0
    for i in range(num_tiles):
        for j in range(num_tiles):
            tile = image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            tile_map[tile_id] = tile
            tile_id += 1
    return tile_map

def reconstruct_image(tile_map, layout):
    patch_size = tile_map[0].shape[0]
    grid_size = len(layout)
    image_out = np.zeros((grid_size * patch_size, grid_size * patch_size), dtype=np.uint8)
    for i, row in enumerate(layout):
        for j, tile_id in enumerate(row):
            image_out[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = tile_map[tile_id]
    return image_out

def display_image(image, title="Image"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def calculate_edge_dissimilarity(tile1, tile2, direction):
    if direction == 'right':
        return np.sum(np.abs(tile1[:, -1] - tile2[:, 0]))
    elif direction == 'left':
        return np.sum(np.abs(tile1[:, 0] - tile2[:, -1]))
    elif direction == 'down':
        return np.sum(np.abs(tile1[-1, :] - tile2[0, :]))
    elif direction == 'up':
        return np.sum(np.abs(tile1[0, :] - tile2[-1, :]))
    return 0

def get_best_matching_tile(parent_tile, available_tiles, direction):
    best_tile_id = -1
    min_dissimilarity = float('inf')
    for tid, tile in available_tiles.items():
        diff = calculate_edge_dissimilarity(parent_tile, tile, direction)
        if diff < min_dissimilarity:
            min_dissimilarity = diff
            best_tile_id = tid
    return best_tile_id

def calculate_layout_cost(layout, tile_map):
    total_cost = 0
    size = len(layout)
    for r in range(size):
        for c in range(size):
            tile_id = layout[r][c]
            if tile_id == -1:
                continue
            current_tile = tile_map[tile_id]
            if c + 1 < size:
                right_id = layout[r][c+1]
                if right_id != -1:
                    total_cost += calculate_edge_dissimilarity(current_tile, tile_map[right_id], 'right')
            if r + 1 < size:
                bottom_id = layout[r+1][c]
                if bottom_id != -1:
                    total_cost += calculate_edge_dissimilarity(current_tile, tile_map[bottom_id], 'down')
    return total_cost

def greedy_bfs_fill(tile_map, start_tile_id):
    size = int(math.sqrt(len(tile_map)))
    layout = [[-1 for _ in range(size)] for _ in range(size)]
    available_ids = set(tile_map.keys())
    if start_tile_id in available_ids:
        layout[0][0] = start_tile_id
        available_ids.remove(start_tile_id)
        queue = deque([(0, 0)])
        visited = set([(0, 0)])
        while queue:
            r, c = queue.popleft()
            parent_tile = tile_map[layout[r][c]]
            for dr, dc, direction in [(1,0,'down'), (0,1,'right'), (-1,0,'up'), (0,-1,'left')]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in visited:
                    candidates = {tid: tile_map[tid] for tid in available_ids}
                    best_tile = get_best_matching_tile(parent_tile, candidates, direction)
                    if best_tile != -1:
                        layout[nr][nc] = best_tile
                        available_ids.remove(best_tile)
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return layout
    return None

def simulated_annealing_refine(initial_layout, tile_map, initial_temp=1000, final_temp=1, alpha=0.995):
    current_layout = copy.deepcopy(initial_layout)
    best_layout = copy.deepcopy(initial_layout)
    current_cost = calculate_layout_cost(current_layout, tile_map)
    best_cost = current_cost
    temp = initial_temp
    steps = 0
    while temp > final_temp:
        steps += 1
        r1, c1 = random.randint(0, len(current_layout)-1), random.randint(0, len(current_layout[0])-1)
        r2, c2 = random.randint(0, len(current_layout)-1), random.randint(0, len(current_layout[0])-1)
        current_layout[r1][c1], current_layout[r2][c2] = current_layout[r2][c2], current_layout[r1][c1]
        new_cost = calculate_layout_cost(current_layout, tile_map)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_layout = copy.deepcopy(current_layout)
        else:
            current_layout[r1][c1], current_layout[r2][c2] = current_layout[r2][c2], current_layout[r1][c1]
        temp *= alpha
    return best_layout, best_cost, steps

def greedy_fill_and_refine_main():
    INPUT_FILE = os.path.join("", "scrambled_lena.mat")
    OUTPUT_DIR = "output"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "solved_lena.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process = psutil.Process()
    image_matrix = load_mat_image(INPUT_FILE)
    if image_matrix is None:
        image_matrix = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))

    tile_map = create_patches(image_matrix)
    scrambled_layout = np.arange(len(tile_map)).reshape(int(math.sqrt(len(tile_map))), -1)
    scrambled_image = reconstruct_image(tile_map, scrambled_layout)
    display_image(scrambled_image, "Initial Scrambled Image")

    start_greedy = time.time()
    best_greedy_layout = None
    min_cost = float("inf")

    for i in tqdm(range(len(tile_map)), desc="Greedy search"):
        layout = greedy_bfs_fill(tile_map, start_tile_id=i)
        if layout is not None:
            cost = calculate_layout_cost(layout, tile_map)
            if cost < min_cost:
                min_cost = cost
                best_greedy_layout = layout

    greedy_time = time.time() - start_greedy
    print(f"Greedy BFS completed in {greedy_time:.2f}s with cost {min_cost:.2f}")

    if best_greedy_layout is not None:
        initial_img = reconstruct_image(tile_map, best_greedy_layout)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "initial_greedy_solution.png"), initial_img)
        display_image(initial_img, "Greedy BFS Solution")

        start_sa = time.time()
        final_layout, final_cost, steps = simulated_annealing_refine(best_greedy_layout, tile_map, initial_temp=50000, alpha=0.998)
        sa_time = time.time() - start_sa

        print(f"Simulated Annealing completed in {sa_time:.2f}s with cost {final_cost:.2f} after {steps} steps")

        total_time = greedy_time + sa_time
        memory_usage = process.memory_info().rss / (1024 * 1024)
        print(f"Total time: {total_time:.2f}s | Memory: {memory_usage:.2f} MB")

        final_img = reconstruct_image(tile_map, final_layout)
        cv2.imwrite(OUTPUT_FILE, final_img)
        display_image(final_img, "Final Solved Puzzle")
    else:
        print("No valid greedy layout found.")

if __name__ == "__main__":
    greedy_fill_and_refine_main()
