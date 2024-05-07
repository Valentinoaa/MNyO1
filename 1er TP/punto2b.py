import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from punto2a import *


def main():

    plt.figure(figsize=(8, 6))

    fst = open_csv('mnyo_mediciones')

    plot_interpolation(fst["c1"], fst["c2"], color='purple')

    scd = open_csv('mnyo_mediciones2')

    plot_interpolation(scd["c1"], scd["c2"], 4, 'g')

    plt.xlabel('Coordenada X1')
    plt.ylabel('Coordenada X2')


    plt.show()

    global f1, f2, g1, g2
    f1 = scipy.interpolate.CubicSpline(np.linspace(1, 10, len(fst["c1"])), fst["c1"])
    f2 = scipy.interpolate.CubicSpline(np.linspace(1, 10, len(scd["c1"])), scd["c1"])

    g1 = scipy.interpolate.CubicSpline(np.linspace(1, 10, len(fst["c2"])), fst["c2"])
    g2 = scipy.interpolate.CubicSpline(np.linspace(1, 10, len(scd["c2"])), scd["c2"])
    

    x, y = newton(f, g, 0.1, 0.5)

    print(f"La intersección de las funciones es: [{f1(x)}, {g2(y)}]")

def f(x, y):
    return f1(x) - f2(y)

def g(x, y):
    return g1(x) - g2(y)


def jacobiano(f1, f2 , x, y, tolerancia=1e-6):
    return np.array(
                    [[(f1(x + tolerancia, y) - f1(x, y)) / tolerancia, (f1(x, y + tolerancia) - f1(x, y)) / tolerancia ], 
                     [(f2(x + tolerancia, y) - f2(x, y)) / tolerancia, (f2(x, y + tolerancia) - f2(x, y)) / tolerancia ]
                    ])

def newton(f1, f2, x0, y0, tol=1e-6, max_iter=1000):

    x = x0
    y = y0

    max_iter = 1000
    tolerancia = 1e-6

    for i in range(max_iter):
        print(jacobiano(f1,f2,x,y))
        j_inv = np.linalg.inv(jacobiano(f1, f2, x, y))

        f = np.array([f1(x, y), f2(x, y)])
        p = np.array([x, y]) - j_inv @ f
        if np.linalg.norm(p - np.array([x, y])) < tolerancia:
            break
        x, y = p

    if i == max_iter - 1:
        print("Maximum number of iterations reached")

    return x, y

    
def interpolate(inter):

    return scipy.interpolate.CubicSpline(inter.index, inter)


if __name__ == "__main__":
    main()


    

