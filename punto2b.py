import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from punto2a import *


def main():

    plt.figure(figsize=(8, 6))

    fst = open_csv('mnyo_mediciones')

    plot_interpolation(fst["c1"], fst["c2"])

    scd = open_csv('mnyo_mediciones2')

    plot_interpolation(scd["c1"], scd["c2"], 4, 'b')

    plt.show()


    f = intersection(fst["c1"], scd["c1"])
    g = intersection(fst["c2"], scd["c2"])

    print(newton(f, g, 0, 0))


    # g = interpolate(dom=np.linspace(scd["c1"].min(), scd["c1"].max(), len(scd["c1"])), y=scd["c1"])
    # h = interpolate(dom=np.linspace(fst["c1"].min(), fst["c1"].max(), len(fst["c1"])), y=fst["c1"])

    for i in np.linspace(scd["c1"].min(), scd["c1"].max(), 100):
        print(f"X : {g(i)} -> F(x) - G(x): {f(i)} ")


def f1(x, dom_x, y, dom_y):

    return interpolate(x=dom_x, y=x) - interpolate(x=dom_x, y=x)


def jacobiano(f1, f2 , x, y, tol=1e-6):

    return np.array[[(f1(x + tol , y)) - f1(x, y) / tol, (f1(x, y +tol)- f1(x, y)) / tol],
                    [(f2(x+tol,y)-f2(x,y)) / tol , (f2(x,y+tol)-f2(x,y))/tol]]


def newton(f1, f2, x0, y0, tol=1e-6):

    x = x0
    y = y0

    while True:

        j = jacobiano(f1, f2, x, y)

        inv_j = np.linalg.inv(j)

        f = np.array([f1(x, y), f2(x, y)])

        x = x - np.dot(inv_j, f)[0]

        y = y - np.dot(inv_j, f)[1]

        if np.linalg.norm(f) < tol:
            break

    return x, y
    

def intersection(fst, scd, jacobian = False):

    dom_fst = np.linspace(fst.min(), fst.max(), len(fst))

    dom_scd = np.linspace(scd.min(), scd.max(), len(scd))

    if not jacobian:
        f = scipy.interpolate.CubicSpline(dom_fst, fst)

        g = scipy.interpolate.CubicSpline(dom_scd, scd)
    
    else:
        f = scipy.interpolate.CubicSpline(dom_fst, fst).derivative()

        g = scipy.interpolate.CubicSpline(dom_scd, scd).derivative()

    return lambda x: f(x) - g(x)        
    

def interpolate(y, dom):

    f = scipy.interpolate.CubicSpline(x=dom, y=y)

    return lambda x: f(x)


if __name__ == "__main__":
    main()


    

