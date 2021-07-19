from os import getloadavg
import numpy as np
import matplotlib.pyplot as plt
import datetime
import dtw
from scipy import interpolate
import os
from skimage.metrics import structural_similarity
from skimage import io
import json

log_dir = "../../logData/"

with open("../positions_walk.txt", 'r') as f:
    positions = list()
    for line in f.readlines():
        pos = np.array([float(x) for x in line.split()[:3]])
        positions.append(pos)

with open("../positions_ref_walk.txt", 'r') as f:
    positions_ref = list()
    for line in f.readlines():
        pos = np.array([float(x) for x in line.split()[:3]])
        positions_ref.append(pos)

def plot(points):
    print(f"{len(points)} positions")
    x_coords = [p[0] for p in points]
    y_coords = [p[2] for p in points]
    plt.scatter(x_coords, y_coords, 1)
    plt.show()

def get_stats(log_name):
    with open(f"{log_dir}{log_name}.txt") as f:
        lines = f.readlines()

    first_timestamp = None

    stats = list()
    for line in lines:
        line.strip()
        fields = [field.split(": ") for field in line.split("; ")]
        parsed = { field[0]: field[1] for field in fields }

        timestamp = datetime.datetime.strptime(parsed["timestamp"].strip(), '%Y-%m-%d_%H-%M-%S.%f')
        if not first_timestamp:
            first_timestamp = timestamp
        time_elapsed = (timestamp - first_timestamp).total_seconds()

        selected_stats = {
            "triangle_count": int(parsed["triangle_count"]),
            "vertex_count": int(parsed["vertex_count"]),
            "textures_count": int(parsed["textures_count"]),
            "fps": float(parsed["fps"]),
            "time_elapsed": time_elapsed
        }

        stats.append(selected_stats)
    return stats

def ssim_from_screenshots(id, id_ref):
    screenshots_dir = "../../screenshots/walk/"
    screenshotsref_dir = "../../screenshots/walk_ref/"
    filepath = os.path.join(screenshots_dir, f"{id}.png")
    ref_filepath = os.path.join(screenshotsref_dir, f"{id_ref}.png")
    image = io.imread(filepath)
    ref_image = io.imread(ref_filepath)
    ssim = structural_similarity(image, ref_image, multichannel=True)
    return ssim

def compute_all_ssims(idx):
    ssims = list()
    for i,id in enumerate(idx):
        print(i, id)
        ssim = ssim_from_screenshots(i, id)
        ssims.append(ssim)
    return ssims

def interpolate_factor(points, factor):
    N = len(points)
    x = np.arange(0, factor*N, factor)
    x_new = np.arange(factor*N-factor+1)
    f = interpolate.interp1d(x, points, axis=0)
    return f(x_new)

def preprocess_timeseries(stats):
    series = np.array([x["time_elapsed"] for x in stats])
    last_time = series[-1]
    return np.divide(series, last_time)

def select_fps(stats):
    return np.array([1/x["fps"] for x in stats])

stats = get_stats("walk")
ref_stats = get_stats("walk_ref")

timeseries = preprocess_timeseries(stats)
timeseries_ref = preprocess_timeseries(ref_stats)

alignment = dtw.dtw(timeseries_ref, timeseries, keep_internals=True)
alignment.plot(type="threeway")
idx = dtw.warp(alignment, index_reference=False)

dataset = {
    "idx": idx.tolist(),
    "ssims": compute_all_ssims(idx)
}

with open("dataset.json", 'w') as f:
    json.dump(dataset, f)

# plt.plot(timeseries[idx])
# plt.plot(timeseries_ref)

# dy = [y-x for x,y in enumerate(idx/3)]
# plt.plot(dy)
# plt.show()
