import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import math
from random import randint

with open('experiment_results.txt') as json_file:
    data = json.load(json_file)


def test_predictor():
    x = list(map(lambda x: [x['average_ssim'], x['average_frame_vertex_count']], data))
    y = list(map(lambda x: x['rating'], data))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = randint(0,10000))

    model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                    ('linear', LinearRegression(fit_intercept=True))])

    model = model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    plt.scatter(test_pred, y_test, s=20)
    plt.xlabel("prediction")
    plt.ylabel("ground truth")
    plt.show()

def vertices_vs_score_plot():
    x = list(map(lambda x: math.log2(x['mesh_vertex_count']), data))
    y = list(map(lambda x: x['rating'], data))
    plt.scatter(x, y, s=20)
    plt.show()

test_predictor()