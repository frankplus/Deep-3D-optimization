import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

with open("dataset.json", 'r') as f:
    dataset = json.load(f)

lod_names = dataset["lod_names"]

data = {lod: list() for lod in lod_names}

for sample in dataset["samples"]:
    distance = np.linalg.norm(np.array(sample["pos"]) - np.array(sample["pos_ref"]))
    ssim = sample["ssim"]
    lod = sample["lod_name"]
    data[lod].append([distance, ssim])

color_dict = {
    lod_names[0]: "red",
    lod_names[1]: "black",
    lod_names[2]: "blue",
}

for lod in lod_names:
    points = np.array(data[lod])
    x = points[:,0]
    y = points[:,1]
    plt.scatter(x, y, s=8, c=color_dict[lod])

    model = LinearRegression().fit(x.reshape((-1, 1)), y.reshape((-1, 1)))
    x_regression = np.arange(0, 0.5, 0.01)
    y_regression = model.predict(x_regression.reshape((-1, 1)))
    plt.plot(x_regression, y_regression, color=color_dict[lod])

plt.xlabel("distance")
plt.ylabel("ssim")
plt.show()