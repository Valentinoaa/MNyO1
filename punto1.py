import scipy
import numpy as np
import matplotlib.pyplot as plt
from functions import f_a, f_b

def plot_a(interval: np.ndarray = np.linspace(-4, 4, 100), grade = range(2, 11)):
    plt.figure(figsize=(8, 6))
    plt.plot(interval, f_a(interval), label= 'Función original', color='black')
    plt.title('Función $f_a(x)$')
    plt.xlabel('x')
    plt.ylabel('$f_a(x)$')
    plt.legend()  # Muestra la leyenda
    plt.show()




def interpolated_a(grade = range(2, 11)):
    interval = np.linspace(-4, 4, 100)
    plt.figure(figsize=(8, 6))
    # plt.plot(np.linspace(-4, 4, 100), x(np.linspace(-4, 4, 100)), label='Polinomio interpolante')
    colors = ['black', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple']  # Lista de 9 colores
    for n, color in zip(grade, colors):
        # Create Lagrange polynomial
        points_a = divide_interval(interval, n)
        x = scipy.interpolate.lagrange(points_a, f_a(points_a))
        # Create subplots
        plt.plot(interval, x(interval), color=color, label=f"Color {color} representa polinomio de Lagrange para {n} puntos")
        
    plt.plot(interval, f_a(interval), label= 'Función original', color='r', linestyle='--')
    plt.title('Función $f_a(x)$')
    plt.xlabel('x')
    plt.ylabel('$f_a(x)$')
    plt.legend()  # Muestra la leyenda
    plt.show()


def calculate_error(grade = range(2, 11)):
        # Error del grafico
    plt.figure(figsize=(8, 6))
    errors = []
    for n in range(2, 11):
        points_a = divide_interval(np.linspace(-4, 4, 100), n)
        g = scipy.interpolate.lagrange(points_a, f_a(points_a))
        errors.append(calculate_error(f_a, g, np.linspace(-4, 4, 100)))

    errors = np.array(errors)  # Convert errors to a numpy array
    print(errors)
    plt.bar(grade, errors)  # Select the first column of errors
    plt.xlabel('Grade')
    plt.ylabel('Error')
    plt.title('Error vs Grade')
    plt.show()

def plot_b(x1: np.ndarray = np.linspace(-1, 1, 100), x2: np.ndarray = np.linspace(-1, 1, 100)):
    x1_plot_b, x2_plot_b = np.meshgrid(x1, x2)
    
    fig = plt.figure(figsize=(3, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_plot_b, x2_plot_b, f_b(x1_plot_b, x2_plot_b), cmap='coolwarm')
    ax.set_title('Función $f_b(x_1, x_2)$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f_b(x_1, x_2)$')
    plt.show()



def interpolated_b(x1: np.ndarray = np.linspace(-1, 1, 9), x2: np.ndarray = np.linspace(-1, 1, 9)):
    grid_x, grid_y = np.meshgrid(x1, x2)
    grid_z = scipy.interpolate.griddata((grid_x.flatten(), grid_y.flatten()), f_b(grid_x, grid_y).flatten(), (grid_x, grid_y), method='linear')
    
    f_b(grid_x, grid_y).flatten()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')
    
    ax.set_title(f'Función interpolada con {len(x1) * len(x2)} puntos $f_b(x_1, x_2)$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f_b(x_1, x_2)$')
    plt.show()


def chebyshev_roots(x_left, x_right, N):
    radius = (x_right - x_left) / 2.
    center = (x_right + x_left) / 2.    
    return center + radius * np.cos(np.pi - np.arange(N+1)*np.pi/N)

def interpolateAchevyshev():
    x = chebyshev_roots(-4, 4, 20)

    f_cheb = f_a(x) 
    f_inter = scipy.interpolate.lagrange(x, f_cheb)

    xx = np.linspace(-4, 4, 300)
    plt.figure(figsize=(8, 6))
    plt.plot(x, f_cheb, label='Chebyshev nodes', marker='o', color='blue', linestyle='None')
    plt.plot(xx, f_inter(xx), label='Interpolated function', color='blue')  
    plt.plot(xx, f_a(xx), label='Original function', color='black')
    plt.title('Interpolated function with Chebyshev nodes')
    plt.xlabel('x')
    plt.ylabel('$f_a(x)$')
    plt.legend()
    plt.show()

def interpolateBchevychev(x1: np.ndarray = np.linspace(-1, 1, 100), x2: np.ndarray = np.linspace(-1, 1, 100)):
    x1_plot_b, x2_plot_b = np.meshgrid(x1, x2)

    x1 = chebyshev_roots(-1, 1, 20)
    x2 = chebyshev_roots(-1, 1, 20)

    # Create meshgrid
    x1, x2 = np.meshgrid(x1, x2)

    # Generate interpolated values
    grid_z = scipy.interpolate.griddata((x1.flatten(), x2.flatten()), f_b(x1, x2).flatten(), (x1, x2), method='linear')

    # Plotting
    fig = plt.figure(figsize=(12, 6))

    # Original function subplot
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x1_plot_b, x2_plot_b, f_b(x1_plot_b, x2_plot_b), edgecolor='none', alpha=0.8, antialiased=True)
    ax.set_title('Original function $f_b(x_1, x_2)$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f_b(x_1, x_2)$')

    # Interpolated function subplot
    ay = fig.add_subplot(122, projection='3d')
    ay.plot_surface(x1, x2, grid_z, cmap='viridis', edgecolor='none', alpha=0.8, antialiased=True)
    ay.set_title('Interpolated function with Chebyshev nodes $f_b(x_1, x_2)$')
    ay.set_xlabel('$x_1$')
    ay.set_ylabel('$x_2$')
    ay.set_zlabel('$f_b(x_1, x_2)$')

    plt.tight_layout()
    plt.show()
    
    plt.show()



def divide_interval(interval: np.ndarray, n: int) -> np.ndarray:
    points = np.linspace(interval[0], interval[-1], num=n)
    return points

def calculate_error(f, g, interval):
    return scipy.integrate.quad(lambda x: abs((f(x) - g(x))), interval[0], interval[-1])

def main():
    plot_a()
    interpolated_a([2, 5, 8, 9, 11])
    plot_b()
    interpolated_b()
    interpolateAchevyshev()
    interpolateBchevychev()


    
if __name__ == "__main__":
    main()