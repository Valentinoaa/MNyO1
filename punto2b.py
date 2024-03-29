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

    f = intersection(fst["c1"], scd["c1"])

    h = intersection(fst["c2"], scd["c2"])

    plt.show()


    start = abs(fst["c2"].min() - scd["c2"].min())

    end = abs(fst["c2"].max() - scd["c2"].max())




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


    

