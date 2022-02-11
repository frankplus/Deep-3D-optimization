import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import math
from random import randint
import numpy as np
from sklearn.metrics import confusion_matrix

with open('experiment_results.txt') as json_file:
    data = json.load(json_file)

def show_confusion_matrix(conf_arr):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = '123456789'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.show()


def test_predictor():
    x = list(map(lambda x: [x['average_ssim'], x['average_frame_vertex_count']], data))
    y = list(map(lambda x: x['rating'], data))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.02, random_state = randint(0,10000))

    # model = Pipeline([('poly', PolynomialFeatures(degree=3)),
    #                 ('linear', LinearRegression(fit_intercept=True))])
    model = LinearRegression(fit_intercept=True)

    model = model.fit(x_train, y_train)

    print(f"coefficients: {model.coef_}")
    print(f"intercept: {model.intercept_}")

    y_pred = np.rint(model.predict(x_train))

    matrix = confusion_matrix(y_train, y_pred, labels=[1,2,3,4,5])
    print(matrix)
    show_confusion_matrix(matrix)

    # plt.scatter(y_pred, y_train, s=5)
    # plt.xlabel("prediction")
    # plt.ylabel("ground truth")
    # plt.xlim(0, 6)
    # plt.ylim(0, 6)
    # plt.show()

def vertices_vs_score_plot():
    x = list(map(lambda x: math.log2(x['mesh_vertex_count']), data))
    y = list(map(lambda x: x['rating'], data))
    plt.scatter(x, y, s=20)
    plt.show()

test_predictor()