import csv
from os import getenv, listdir
from os.path import isfile, join
import re
from matplotlib.pyplot import axis
import onnx
import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import math
import json

data_path = './data/'
files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
files = map(lambda name: name.split('.'), files)
files = filter(lambda x: x[1] == 'csv', files)
files = map(lambda x: x[0].split('_')[1], files)
files = list(files)

models = {
    "temple000": ["Pagoda_05", "Pagoda_20", "Pagoda_50"],
    "dragon000": ["Drago_01", "Drago_05", "Drago_10"],
    "Owl_high000": ["Gufo_01", "Gufo_10", "Gufo_20"],
    "xyzrgb_statuette000": ["Statua_01", "Statua_05", "Statua_10"],
    "notredame000": ["Notredame_10", "Notredame_20", "Notredame_50"]
}

vertex_count_map = {
    "Pagoda_05": 48365, 
    "Pagoda_20": 144839, 
    "Pagoda_50": 297499,
    "Drago_01": 49106, 
    "Drago_05": 457367, 
    "Drago_10": 1132209,
    "Gufo_01": 22015, 
    "Gufo_10": 359912, 
    "Gufo_20": 895441,
    "Statua_01": 83764, 
    "Statua_05": 767210, 
    "Statua_10": 1788979,
    "Notredame_10": 330260, 
    "Notredame_20": 582147, 
    "Notredame_50": 1690643
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def open_user_csv(userid):
    with open(f"data/fileseval_{userid}.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        first = True
        data = list()
        for row in reader:
            if first: 
                first = False
                continue
            parsed = row[0].split(',')
            data.append({
                            'UserID': parsed[0], 
                            'DisplayID': parsed[1], 
                            'Rating': int(parsed[2]), 
                            'StartTime': int(parsed[3]), 
                            'EndTime': int(parsed[4])
                        })

        return data

def open_movement_log(userid, displayid):
    with open(f"data/Log movimento/fileslog_{displayid}_{userid}.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        first = True
        data = list()
        for row in reader:
            if first: 
                first = False
                continue
            s = ''.join(row)
            pattern = '([0-9]+),"\((-?\d*\.?\d*),(-?\d*\.?\d*),(-?\d*\.?\d*)\)","\((-?\d*\.?\d*),(-?\d*\.?\d*),(-?\d*\.?\d*),(-?\d*\.?\d*)\)",(-?\d*\.?\d*),(U\d+),([a-zA-Z0-9_]*)'
            groups = re.match(pattern, s).groups()
            parsed = {'Timestamp': int(groups[0]), 
                    'Position': list(map(lambda x: float(x), groups[1:4])), 
                    'Rotation': list(map(lambda x: float(x), groups[4:8])), 
                    'fps': float(groups[8]), 
                    'UserID': groups[9],
                    'DisplayID': groups[10]}
            
            # shift y coordinate by -1 because of incompatibility between logs and trained model
            parsed['Position'][1] -=1 

            data.append(parsed)
        return data

def load_projections(models, device):
    # loader uses the transforms function that comes with torchvision
    loader = transforms.Compose([
        transforms.ToTensor()])  

    # Enter the picture address
    # Return tensor variable
    def image_loader(image_name):
        image = Image.open(image_name).convert('RGB').split()[0]
        image = loader(image)
        return image.to(device, torch.float)

    def get_path(model, plane):
        projections_dir = "projections/"
        return f"{projections_dir}{model}_{plane}.png"

    projections = dict()
    for model in models:
        planes = [image_loader(get_path(model, plane)) for plane in ["H", "V", "L"]]
        tensor = torch.cat(planes, 0)
        projections[model] = tensor
    return projections

def get_projections_by_displayid(displayid, projections):
    for key in models:
        if displayid in models[key]:
            return projections[key]

def compute_cnn_features(cnn_session, projections):
    proj = np.expand_dims(projections, 0)
    batch = np.concatenate((proj, np.zeros((3,3,256,256))))
    outputs = cnn_session.run(None, {'input': batch.astype(np.float32)})
    return outputs[0][0]

def get_target_sample(logs, curr_timestamp, time_interval = 100):
    next_timestamp = curr_timestamp + time_interval
    for sample in logs:
        if sample['Timestamp'] >= next_timestamp:
            return sample
    return None

def calculate_average_indices(logs, mesh_vertex_count, cnn_features, ffn_session, dynamic_ssim=True):
    output_sequence = list()
    sum = 0
    for sample in logs:
        if dynamic_ssim:
            target_sample = get_target_sample(logs, sample['Timestamp'])
            if not target_sample:
                continue
        ref_position = np.array(sample['Position'])

        if dynamic_ssim:
            target_position = np.array(target_sample['Position'])
        else:
            target_position = ref_position

        norm_mesh_vertex_count = [math.log2(mesh_vertex_count) / 30.0]
        ffn_input = np.concatenate((target_position, ref_position, norm_mesh_vertex_count, cnn_features))
        ffn_input = np.expand_dims(ffn_input, 0)
        batch = np.concatenate((ffn_input, np.zeros((3,15))))
        outputs = ffn_session.run(None, {'input': batch.astype(np.float32)})
        output_sequence.append(outputs[0][0])

        sum += np.linalg.norm(ref_position - target_position)
    
    print("average distance: " + str(sum / len(output_sequence)))
    return np.average(output_sequence, axis=0)


cnnmodel = onnx.load("ONNX/cnn.onnx")
ffnmodel = onnx.load("ONNX/ffn.onnx")
onnx.checker.check_model(cnnmodel)
onnx.checker.check_model(ffnmodel)
cnnsession = ort.InferenceSession("ONNX/cnn.onnx")
ffnsession = ort.InferenceSession("ONNX/ffn.onnx")

all_projections = load_projections(models, device)


experiments = list()
for file in files:
    for data in open_user_csv(file):
        if data['DisplayID'] in ['Specchio_LQ', 'Specchio_MQ', 'Specchio_HQ']:
            continue

        logs = open_movement_log(data['UserID'], data['DisplayID'])

        proj = get_projections_by_displayid(data['DisplayID'], all_projections)
        cnn_features = compute_cnn_features(cnnsession, proj)

        average_indices = calculate_average_indices(logs, vertex_count_map[data['DisplayID']], cnn_features, ffnsession, dynamic_ssim=True)
        average_indices = average_indices.astype(float)
        results = {
                    'average_ssim': average_indices[0], 
                    'average_frame_vertex_count': average_indices[1], 
                    'rating': data['Rating'],
                    'mesh_vertex_count': vertex_count_map[data['DisplayID']]
                }
        print(results)
        experiments.append(results)

with open('experiment_results.txt', 'w') as outfile:
    json.dump(experiments, outfile)