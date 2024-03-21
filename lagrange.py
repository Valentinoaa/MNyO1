import scipy
import numpy as np
import matplotlib.pyplot as plt
from functions import f_a, f_b


def plot_a(interval: np.ndarray = np.linspace(-4, 4, 1000)):
    plt.figure(figsize=(8, 6))
    
    for n in range(2, 6):
        points_a = divide_interval(interval, n)
        x = scipy.interpolate.lagrange(points_a, f_a(points_a))
        plt.plot(interval, x(interval), label=f'n = {n}')
        plt.plot(points_a, f_a(points_a), 'o', label='Puntos de interpolación')
        plt.plot(np.linspace(-4, 4, 100), x(np.linspace(-4, 4, 100)), label='Polinomio interpolante')
    
    plt.plot(interval, f_a(interval))
    plt.title('Función $f_a(x)$')
    plt.xlabel('x')
    plt.ylabel('$f_a(x)$')
    plt.show()

def plot_b(x1: np.ndarray = np.linspace(-1, 1, 100), x2: np.ndarray = np.linspace(-1, 1, 100)):
    x1_plot_b, x2_plot_b = np.meshgrid(x1, x2)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_plot_b, x2_plot_b, f_b(x1_plot_b, x2_plot_b))
    ax.set_title('Función $f_b(x_1, x_2)$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f_b(x_1, x_2)$')
    plt.show()



def interpolated_b(x1: np.ndarray = np.linspace(-1, 1, 100), x2: np.ndarray = np.linspace(-1, 1, 100)):
    grid_x, grid_y = np.meshgrid(x1, x2)
    grid_z = scipy.interpolate.griddata((grid_x.flatten(), grid_y.flatten()), f_b(grid_x, grid_y).flatten(), (grid_x, grid_y), method='linear')
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z)
    ax.set_title('Función interpolada con 10000 puntos $f_b(x_1, x_2)$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f_b(x_1, x_2)$')
    plt.show()


def divide_interval(interval: np.ndarray, n: int) -> np.ndarray:
    points = np.linspace(interval[0], interval[-1], num=n)
    return points

def main():
    plot_a()
    plot_b()
    
    interpolated_b()
    
if __name__ == "__main__":
    main()