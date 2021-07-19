import json
import matplotlib.pyplot as plt
import numpy as np

with open("dataset.json", 'r') as f:
    dataset = json.load(f)
lod_names = sorted(dataset["data"].keys())

def select_ssim_data(lod_name):
    return [x["ssim"] for x in dataset["data"][lod_name]]

def show_2d_ssim_plot():
    x = list(map(lambda pos: pos[0], dataset["positions"]))
    y = list(map(lambda pos: pos[2], dataset["positions"]))
    z = select_ssim_data(lod_names[0])
    plt.scatter(x, y, c=z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def show_sorted_data():
    for lod in lod_names:
        data = sorted(select_ssim_data(lod))
        plt.plot(data)
    plt.show()

def plot_fps_vs_ssim():
    lod = lod_names[-1]
    x_points = list()
    y_points = list()
    for sample in dataset["data"][lod]:
        x_points.append(sample["ssim"])
        y_points.append(sample["fps"])

    plt.scatter(x_points, y_points)
    plt.show()

    plt.hist(y_points, 50)
    plt.show()

def plot_ssim_vs_distance():
    for lod in lod_names:
        x_axis = [np.linalg.norm(x) for x in dataset["positions"]]
        y_axis = select_ssim_data(lod)
        plt.scatter(x_axis, y_axis)
    plt.xlabel("distance from center")
    plt.ylabel("ssim")
    plt.show()

show_2d_ssim_plot()
plot_ssim_vs_distance()