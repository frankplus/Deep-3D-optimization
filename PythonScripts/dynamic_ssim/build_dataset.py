from posixpath import dirname
import numpy as np
import os
from skimage.metrics import structural_similarity
from skimage import io
import matplotlib.pyplot as plt
from skimage.util import compare_images
import json
import random

log_dir = "../../logData/noscreenshot/"
screenshots_path = "../../screenshots/"
ref_dir = "Owl_high"

# list all directories inside screenshots_path
dirpath, dirnames, filenames = next(os.walk(screenshots_path))
dirnames = sorted(filter(lambda dir: dir != ref_dir, dirnames))

def load_positions():
    with open("../positions.txt", 'r') as f:
        positions = list()
        for line in f.readlines():
            pos = np.array([float(x) for x in line.split()[:3]])
            positions.append(pos)
    return positions

def generate_samples(positions):
    MAX_DISTANCE = 0.2
    MAX_NUM_PAIRS = 10000
    samples = list()
    for i in range(len(positions)):
        for j in range(i,len(positions)):
            pair = [i,j] 
            random.shuffle(pair)
            if np.linalg.norm(positions[pair[0]] - positions[pair[1]]) < MAX_DISTANCE:
                for lod_name in dirnames:
                    samples.append({
                        "i": i,
                        "j": j,
                        "lod_name": lod_name
                    })
    random.shuffle(samples)
    if len(samples) > MAX_NUM_PAIRS:
        samples = samples[:MAX_NUM_PAIRS]
    return samples

def compute_ssim_from_pair(i, j, dir, ref_dir):
    filepath_i = os.path.join(dir, f"{i}.png")
    filepath_j = os.path.join(ref_dir, f"{j}.png")
    image_i = io.imread(filepath_i)
    image_j = io.imread(filepath_j)
    return structural_similarity(image_i, image_j, multichannel=True)

def diff_image_from_pair(i, j, dir, ref_dir):
    filepath_i = os.path.join(dir, f"{i}.png")
    filepath_j = os.path.join(ref_dir, f"{j}.png")
    image_i = io.imread(filepath_i)
    image_j = io.imread(filepath_j)
    return compare_images(image_i, image_j, method='diff')


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

def generate_dataset(samples):
    all_stats = dict()
    for lod in dirnames:
        all_stats[lod] = get_lod_stats(lod)

    dataset = list()
    for i,sample in enumerate(samples):
        print(i)
        path = screenshots_path + sample["lod_name"]
        path_ref = screenshots_path + ref_dir
        ssim = compute_ssim_from_pair(sample["i"], sample["j"], path, path_ref)
        sample = {
            "pos": positions[sample["i"]].tolist(),
            "pos_ref": positions[sample["j"]].tolist(),
            "lod_name": sample["lod_name"],
            "ssim": ssim,
            "fps": all_stats[sample["lod_name"]][sample["i"]]["fps"]
        }
        dataset.append(sample)
    return dataset

    
def plot_image_grid(images, ncols=None, cmap='gray'):
    '''Plot a grid of images'''
    if not ncols:
        factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    axes = axes.flatten()[:len(imgs)]
    for img, ax in zip(imgs, axes.flatten()): 
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.imshow(img, cmap=cmap)
    plt.show()


positions = load_positions()
samples = generate_samples(positions)
print(f"{len(samples)} pairs generated")

dataset = {
    "lod_names": dirnames,
    "samples": generate_dataset(samples)
}

# save
with open("dataset.json", 'w') as f:
    json.dump(dataset, f)