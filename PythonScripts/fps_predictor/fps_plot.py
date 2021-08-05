import numpy as np
import os
import matplotlib.pyplot as plt
import math

log_dir = "../../logData/noscreenshot/"

dirpath, dirnames, filenames = next(os.walk(log_dir))
lod_names = map(lambda x: x.split('.')[0], filenames)
lod_names = filter(lambda x: x != "", lod_names)
lod_names = sorted(lod_names)
print(lod_names)

def load_positions():
    with open("../positions.txt", 'r') as f:
        positions = list()
        for line in f.readlines():
            pos = np.array([float(x) for x in line.split()[:3]])
            positions.append(pos)
    return positions

def get_lod_stats(lod_name):
    with open(f"{log_dir}{lod_name}.txt") as f:
        lines = f.readlines()

    positions = load_positions()

    stats = list()
    for i, line in enumerate(lines):
        line.strip()
        fields = [field.split(": ") for field in line.split("; ")]
        parsed = { field[0]: field[1] for field in fields }

        selected_stats = {
            "position": positions[i],
            "triangle_count": int(parsed["triangle_count"]),
            "vertex_count": int(parsed["vertex_count"]),
            "textures_count": int(parsed["textures_count"]),
            "fps": float(parsed["fps"])
        }

        stats.append(selected_stats)
    return stats

def show_2d_fps_plot(stats):
    x = list(map(lambda x: x["position"][0], stats))
    y = list(map(lambda x: x["position"][2], stats))
    z = list(map(lambda x: x["fps"], stats))
    plt.scatter(x, y, c=z, vmin=0, vmax=30)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def show_distance_fps_plot(stats):
    dist = list(map(lambda x: np.linalg.norm(x["position"]), stats))
    fps = list(map(lambda x: x["fps"], stats))
    plt.scatter(dist, fps)
    plt.ylim(0,40)
    plt.xlabel("distance")
    plt.ylabel("fps")
    plt.show()

def show_fps_hists():
    fig, ax = plt.subplots(len(lod_names))
    for i,lod in enumerate(lod_names):
        stats = get_lod_stats(lod)
        data = list(map(lambda x: x["fps"], stats))
        ax[i].set_title(lod)
        ax[i].hist(data, bins=50, range=(0,200))
    plt.show()

def show_fps_plots():

    def movingvariance(data, window_size):
        moving_variance = list()
        for i in range(len(data)-window_size):
            variance = np.var(data[i:i+window_size])
            moving_variance.append(variance)
        return moving_variance

    def movingaverage(data, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(data, window, 'valid')

    for i,lod in enumerate(lod_names):
        stats = get_lod_stats(lod)
        data = list(map(lambda x: x["fps"], stats))
        data = movingaverage(data, 32)
        plt.plot(data, label=lod)

    plt.legend()
    plt.show()

def print_averages():
    keys = ["fps", "triangle_count", "vertex_count"]
    for key in keys:
        for lod in lod_names:
            stats = get_lod_stats(lod)
            data = list(map(lambda x: x[key], stats))
            average = sum(data) / len(data)
            print(f"{lod} average {key}: {average}")

def vertex_fps_plot():
    for lod in lod_names:
        stats = get_lod_stats(lod)
        vertex_count = list(map(lambda x: math.log2(x["vertex_count"]), stats))
        fps = list(map(lambda x: x["fps"], stats))
        plt.scatter(vertex_count, fps, s=4)
    plt.xlabel("log2(vertex count)")
    plt.ylabel("fps")
    plt.show()
        

print_averages()
vertex_fps_plot()
# show_fps_plots()
# show_fps_hists()
# stats = get_lod_stats(lod_names[0])
# show_2d_fps_plot(stats)
# show_distance_fps_plot(stats)