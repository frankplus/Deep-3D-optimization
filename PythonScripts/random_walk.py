from random import random
from math import pi, cos, sin
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.npyio import save
from scipy import interpolate

def generate_random_walk():
    pr=0.2
    qr=0.7
    pt=0.5
    qt=0.1

    delta_r=0.1
    delta_t=0.01

    points = list()

    for _ in range(10):
        theta0 = random()*pi/3
        r0=10

        r=r0
        theta=theta0

        for c in range(1,100):
            
            if c%100 == 99:
                delta_r=0.1*(random()-0.5)
                delta_t=0.05*(random()-0.5)
            
            if r<3:
                pr=0.7 
                qr=0.2
            else:
                pr=0.2
                qr=0.7

            z = r*cos(theta)
            x = r*sin(theta)
            y = 1.0

            points.append([x,y,z])
            
            flg = random()
            if (flg < pr) and (r < 12):
                r = r + delta_r
            else:
                flg = random()
                if (flg < qr) and (r > 1):
                    r = r - delta_r
            
            flg = random()
            if flg < pt:
                theta += delta_t
            else:
                flg=random()
                if flg < qt:
                    theta -= delta_t
        
    return points
        

def plot(points):
    print(f"{len(points)} positions")
    x_coords = [p[0] for p in points]
    y_coords = [p[2] for p in points]
    plt.scatter(x_coords, y_coords, 1)
    plt.show()

def save_path(points, filename):
    with open(filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]} 0 0 0\n")

def generate_ref_path(points, factor):
    N = len(points)
    x = np.arange(0, factor*N, factor)
    x_new = np.arange(factor*N-factor+1)
    f = interpolate.interp1d(x, points, axis=0)
    return f(x_new)


if __name__ == "__main__":
    points = generate_random_walk()
    save_path(points, "positions_walk.txt")
    ref_points = generate_ref_path(points, 3)
    save_path(ref_points, "positions_ref_walk.txt")
    plot(points)
    plot(ref_points)