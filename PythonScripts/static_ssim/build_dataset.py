from ssl import ALERT_DESCRIPTION_RECORD_OVERFLOW
from skimage.metrics import structural_similarity
from skimage import io
import os
import matplotlib.pyplot as plt
import json

log_dir = "../../logData/"
screenshots_path = "../../screenshots"
ref_dir = ""

# list all directories inside screenshots_path
dirpath, dirnames, filenames = next(os.walk(screenshots_path))
dirnames = sorted(filter(lambda dir: dir != ref_dir, dirnames))


def ssim_from_screenshots(screenshots_dir, screenshotsref_dir):
    list_files = filter(lambda x: x.endswith(".png"), os.listdir(screenshots_dir))
    list_files = sorted(list_files, key=lambda filename: int(filename.split('.')[0]))
    all_ssim = list()
    for filename in list_files:
        print(filename)
        filepath = os.path.join(screenshots_dir, filename)
        ref_filepath = os.path.join(screenshotsref_dir, filename)
        image = io.imread(filepath)
        ref_image = io.imread(ref_filepath)
        ssim = structural_similarity(image, ref_image, multichannel=True)
        all_ssim.append(ssim)
    return all_ssim

def get_lod_stats(lod_name):
    with open(f"{log_dir}{lod_name}.txt") as f:
        lines = f.readlines()

    stats = list()
    for line in lines:
        line.strip()
        fields = [field.split(": ") for field in line.split("; ")]
        parsed = { field[0]: field[1] for field in fields }

        selected_stats = {
            "triangle_count": int(parsed["triangle_count"]),
            "vertex_count": int(parsed["vertex_count"]),
            "textures_count": int(parsed["textures_count"]),
            "fps": float(parsed["fps"])
        }

        stats.append(selected_stats)
    return stats


def generate_lod_data(lod_name):
    ssims = ssim_from_screenshots(screenshots_path + lod_name, screenshots_path + ref_dir)
    stats = get_lod_stats(lod_name)
    for i, ssim in enumerate(ssims):
        stats[i]["ssim"] = ssim
    return stats

data = dict()
for dir in dirnames:
    print(dir)
    data[dir] = generate_lod_data(dir)

# read positions
with open("../positions.txt", 'r') as f:
    positions = list()
    for line in f.readlines():
        pos = [float(x) for x in line.split()[:3]]
        positions.append(pos)

dataset = {"positions": positions, "data": data}

# save
with open("dataset.json", 'w') as f:
    json.dump(dataset, f)