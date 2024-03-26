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


if __name__ == "__main__":
    main()


    

