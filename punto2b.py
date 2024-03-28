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


    f = intersection(fst["c2"], scd["c2"])

    print(f(3))
    plt.show()


def intersection(fst, scd):
    dom_fst = np.linspace(0, 10, len(fst))
    dom_scd = np.linspace(0, 10, len(scd))

    f = scipy.interpolate.CubicSpline(dom_fst, fst)
    g = scipy.interpolate.CubicSpline(dom_scd, scd)

    return lambda x: f(x) - g(x)        
    

def interpolate(y, dom):

    f = scipy.interpolate.CubicSpline(dom, y)

    return f(dom)


if __name__ == "__main__":
    main()


    

