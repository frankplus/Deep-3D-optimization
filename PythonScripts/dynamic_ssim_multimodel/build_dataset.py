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

# a dictionary of LOD levels for each 3d model. The key is the reference LOD, value is a list of LOD levels
train_models = {
    "dragon000": ["dragon001", "dragon002", "dragon003"],
    "notredame000": ["notredame001", "notredame002", "notredame003"],
    "Owl_high000": ["Owl_high001", "Owl_high002", "Owl_high003"],
    "temple000": ["temple001", "temple002", "temple003"],
    "xyzrgb_statuette000": ["xyzrgb_statuette001", "xyzrgb_statuette002", "xyzrgb_statuette003"],
    "DeathValley Mesh Output": ["DeathValley Mesh Output2", "DeathValley Mesh Output3", "DeathValley Mesh Output4"],
    "LacockAbbey02": ["LacockAbbey02b", "LacockAbbey02c", "LacockAbbey02d"]
}

test_models = {
    "Lidded-Ewer0": ["Lidded-Ewer1", "Lidded-Ewer2", "Lidded-Ewer3"]
}

def load_positions():
    with open("../positions.txt", 'r') as f:
        positions = list()
        for line in f.readlines():
            pos = np.array([float(x) for x in line.split()[:3]])
            positions.append(pos)
    return positions

def generate_samples(positions, models, max_num_pairs):
    MAX_DISTANCE = 0.2
    samples = list()
    for i in range(len(positions)):
        for j in range(i,len(positions)):
            pair = [i,j] 
            random.shuffle(pair)
            if np.linalg.norm(positions[pair[0]] - positions[pair[1]]) > MAX_DISTANCE:
                continue

            for model in models:
                for lod_name in models[model]:
                    samples.append({
                        "i": i,
                        "j": j,
                        "model": model,
                        "lod_name": lod_name
                    })

    random.shuffle(samples)
    if len(samples) > max_num_pairs:
        samples = samples[:max_num_pairs]
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

def generate_dataset(samples, models):
    all_stats = dict()
    for model in models:
        for lod_name in models[model]:
            all_stats[lod_name] = get_lod_stats(lod_name)

    dataset = list()
    for i,sample in enumerate(samples):
        print(i)
        path = screenshots_path + sample["lod_name"]
        path_ref = screenshots_path + sample["model"]
        ssim = compute_ssim_from_pair(sample["i"], sample["j"], path, path_ref)
        sample = {
            "pos": positions[sample["i"]].tolist(),
            "pos_ref": positions[sample["j"]].tolist(),
            "model": sample["model"],
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
samples = generate_samples(positions, train_models, max_num_pairs=10000)
print(f"{len(samples)} training pairs generated")

test_samples = generate_samples(positions, test_models, max_num_pairs=1000)
print(f"{len(test_samples)} test pairs generated")

dataset = {
    "models": train_models,
    "samples": generate_dataset(samples, train_models),
    "test_models": test_models,
    "test_samples": generate_dataset(test_samples, test_models)
}

# save
with open("dataset.json", 'w') as f:
    json.dump(dataset, f)