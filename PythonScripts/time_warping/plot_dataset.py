import json
import matplotlib.pyplot as plt
import numpy as np

with open("dataset.json", 'r') as f:
    dataset = json.load(f)

plt.plot(dataset["ssims"])
plt.show()

dy = [y-x for x,y in enumerate(np.array(dataset["idx"])/3)]
plt.plot(dy)
plt.show()