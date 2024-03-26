import pandas as pd
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

def main():

    inter = open_csv('mnyo_mediciones', 'c1', 'c2')

    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')


    # plt.plot(inter['column1'], inter['column2'], 'b')    


    plot_interpolation(inter['c1'], inter['c2'])

    plot_ground_truth()

    plt.legend(['Interpolación', 'Función original'])

    plt.show()

def open_csv(file: str = 'mnyo_meduciones', column1: str = 'c1', column2: str = 'c2' ) -> pd.DataFrame:
    inter = pd.read_csv(f'Mediciones/{file}.csv' , header=None, sep=' ')

    inter.columns = [column1, column2]
    return inter


def plot_interpolation(y1: np.ndarray = np.linspace(-1, 1, 100), y2: np.ndarray = np.linspace(-1, 1, 100), points=10, color='r', kind='cubic', linestyle='None'):
    x = np.linspace(0, points, points)
    grid = scipy.interpolate.interp1d(x, [y1, y2], kind=kind)
    plt.plot(grid(np.linspace(0, points, 100))[0], grid(np.linspace(0, points, 100))[1], color, linestyle=linestyle)

def plot_ground_truth():
    df = open_csv('mnyo_ground_truth')
    plt.plot(df['c1'], df['c2'], 'b')





if __name__ == "__main__":
    main()
    