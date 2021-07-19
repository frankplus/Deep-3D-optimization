import numpy as np
import matplotlib.pyplot as plt
from tsp_solver.greedy import solve_tsp
from scipy.spatial import distance_matrix

def generate_positions(num_samples):
    xs = np.random.uniform(-1,1,num_samples)
    ys = np.random.uniform(-1,1,num_samples)
    zs = np.random.uniform(-1,1,num_samples)
    positions = np.dstack((xs, ys, zs))[0]
    return positions

def save_walk(positions, filename):
    dist_matrix = distance_matrix(positions, positions)
    route = solve_tsp(dist_matrix)

    # Reorder the positions matrix by route order
    positions = np.array([positions[route[i]] for i in range(len(route))])

    # Plot the positions.
    plt.scatter(positions[:,0],positions[:,2])
    # Plot the path.
    # plt.plot(positions[:,0],positions[:,2])
    plt.show()

    print(f"{len(positions)} positions")
    with open(filename, 'w') as f:
        for position in positions:
            f.write(f"{position[0]} {position[1]} {position[2]} 0 0 0\n")

def generate_clustered_positions(num_seed_positions, cluster_size=4):
    seed_positions = generate_positions(num_seed_positions)
    res_positions = seed_positions.tolist()
    for seed in seed_positions:
        for i in range(cluster_size):
            v = np.random.uniform(-1, 1, 3)
            v /= np.linalg.norm(v)
            r = np.random.uniform(0, 0.2)
            v *= r
            pos = seed + v
            res_positions.append(pos)
    return res_positions

    

# generate_positions(100, "positions.txt")
positions = generate_clustered_positions(50)
save_walk(positions, "positions.txt")