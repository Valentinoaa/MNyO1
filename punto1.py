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




def lagrange(grade = range(2, 11)):
    interval = np.linspace(-4, 4, 100)
    plt.figure(figsize=(8, 6))
    # plt.plot(np.linspace(-4, 4, 100), x(np.linspace(-4, 4, 100)), label='Polinomio interpolante')
    colors = ['purple', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'black']  # Lista de 9 colores

    # Lagrange
    for n, color in zip(grade, colors):
        # Create Lagrange polynomial
        points_a = divide_interval(interval, n)
        x = scipy.interpolate.lagrange(points_a, f_a(points_a))
        # Create subplots
        plt.plot(interval, x(interval), color=color, label=f"Polinomio de Lagrange con {n} puntos")
        plt.plot(points_a, f_a(points_a), 'o', color=color)


        
    plt.plot(interval, f_a(interval), label= 'Función original', color='black', linestyle='--')
    #plt.title('Función $f_a(x)$ interpolada con Lagrange')
    plt.xlabel('x')
    plt.ylabel('$f_a(x)$')
    plt.legend()  # Muestra la leyenda
    plt.grid()
    #plt.savefig('lagrange.png')
    plt.show()


def splines(grade = range(2, 11)):
    interval = np.linspace(-4, 4, 100)
    plt.figure(figsize=(8, 6))
    plt.plot(interval, f_a(interval), label='Función original', color='black', linestyle='--')
    colors = ['purple', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'black']  # Lista de 9 colores


    # Spline
    for n, color in zip(grade, colors):
        points_a = divide_interval(interval, n)
        x = scipy.interpolate.CubicSpline(points_a, f_a(points_a))
        plt.plot()
        
        plt.plot(interval, x(interval),color=color,label=f"Spline cúbico con {n} puntos")
        plt.plot(points_a, f_a(points_a), 'o', color=color)

    #plt.title('Función $f_a(x)$ interpolada con Splines')
    plt.xlabel('x')
    plt.ylabel('$f_a(x)$')
    plt.legend()  # Muestra la leyenda
    plt.grid()
    #plt.savefig('splines.png')
    plt.show()

def lagrange_splines_equi(grade=8, interval=np.linspace(-4, 4, 1000), mode='normal'):
    """
    Plot the Lagrange interpolation and cubic spline of a given function.

    Parameters:
    - grade (int): The number of points to use for interpolation. Default is 12.
    - interval (array-like): The interval over which to plot the function. Default is np.linspace(-4, 4, 100).
    - mode (str): The mode of the plot. Default is 'normal'.

    Returns:
    None
    """

    plt.figure(figsize=(8, 6))
    plt.plot(interval, f_a(interval), label='Función original', color='black', linestyle='--')
    points_a = divide_interval(interval, grade)
    x = scipy.interpolate.lagrange(points_a, f_a(points_a))
    plt.plot(interval, x(interval), label='Polinomio de Lagrange', color='g')
    x2= scipy.interpolate.CubicSpline(points_a, f_a(points_a))
    plt.plot(interval, x2(interval), label='Spline cúbico', color='purple')

    plt.plot(points_a, f_a(points_a), 'o', color='black', label='Puntos de interpolación')
    #plt.title(f'Función $f_a(x)$ con {grade} puntos')
    plt.xlabel('x')
    plt.ylabel('$f_a(x)$')
    plt.legend()  # Muestra la leyenda
    plt.grid()
    plt.savefig('lagrange_splines_equidistantes.png')
    plt.show()

def errores_absolutos_equis(grade = 12):
    interval = np.linspace(-4, 4, 1000)
    points_a = divide_interval(interval, grade)
    g1 = scipy.interpolate.lagrange(points_a, f_a(points_a))

    g2 = scipy.interpolate.CubicSpline(points_a, f_a(points_a))
    
    plt.figure(figsize=(8, 6))
    plt.plot(interval, abs(f_a(interval)-g1(interval)), label='Error de Lagrange', color='g')
    plt.plot(interval, abs(f_a(interval)-g2(interval)), label='Error de Splines', color='purple')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Errores absolutos de Lagrange y Splines en puntos equidistantes')
    plt.legend()
    plt.grid()
    plt.show()
def errores_absolutos_chebyshev(grade = 12):
    interval = np.linspace(-4, 4, 1000)
    points_a = chebyshev_roots(-4, 4, grade)
    g1 = scipy.interpolate.lagrange(points_a, f_a(points_a))

    g2 = scipy.interpolate.CubicSpline(points_a, f_a(points_a))
    
    plt.figure(figsize=(8, 6))
    plt.plot(interval, abs(f_a(interval)-g1(interval)), label='Error de Lagrange', color='g')
    plt.plot(interval, abs(f_a(interval)-g2(interval)), label='Error de Splines', color='purple')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Errores absolutos de Lagrange y Splines en puntos de Chebyshev')
    plt.legend()
    plt.grid()
    plt.show()
        
    


def calculate_errors(grade = range(2, 14)):
    # Error de Lagranfe
    errors = []
    for n in grade:
        points_a = divide_interval(np.linspace(-4, 4, 100), n)
        g = scipy.interpolate.lagrange(points_a, f_a(points_a))
        lagrange_error = calculate_error(f_a, g, np.linspace(-4, 4, 100))

        h = scipy.interpolate.CubicSpline(points_a, f_a(points_a))
        spline_error = calculate_error(f_a, h, np.linspace(-4, 4, 100))

        errors.append((lagrange_error, spline_error))
    
    # Plotting the errors
    lagrange_errors, spline_errors = zip(*errors)
    x = np.arange(len(grade))

    plt.figure(figsize=(8, 6))
    plt.bar(x, lagrange_errors, width=0.4, label='Error de Lagrange ', color='g')
    plt.bar(x + 0.4, spline_errors, width=0.4, label='Error de Splines ', color='purple')
    plt.xlabel('Grado')
    plt.ylabel('Error')
    plt.title('Error vs Grado')
    plt.xticks(x, grade)
    plt.legend()

    plt.show()
def calculate_errors_cheb(grade = range(2, 14)):
    
    # Error de Lagranfe
    errors = []
    for n in grade:
        points_a = chebyshev_roots(-4, 4, n)
        g = scipy.interpolate.lagrange(points_a, f_a(points_a))
        lagrange_error = calculate_error(f_a, g, np.linspace(-4, 4, 100))

        h = scipy.interpolate.CubicSpline(points_a, f_a(points_a))
        spline_error = calculate_error(f_a, h, np.linspace(-4, 4, 100))

        errors.append((lagrange_error, spline_error))
    
    # Plotting the errors
    lagrange_errors, spline_errors = zip(*errors)
    x = np.arange(len(grade))

    plt.figure(figsize=(8, 6))
    plt.bar(x, lagrange_errors, width=0.4, label='Error de Lagrange ', color='g')
    plt.bar(x + 0.4, spline_errors, width=0.4, label='Error de Splines ', color='purple')
    plt.xlabel('Grado')
    plt.ylabel('Error con Chebyshev')
    plt.title('Error vs Grado')
    plt.xticks(x, grade)
    plt.legend()

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
    grid_z = scipy.interpolate.griddata((grid_x.flatten(), grid_y.flatten()), f_b(grid_x, grid_y).flatten(), (grid_x, grid_y), method='cubic')
    
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
    """
    Generates the Chebyshev roots within the interval [x_left, x_right].

    Parameters:
    x_left (float): The left endpoint of the interval.
    x_right (float): The right endpoint of the interval.
    N (int): The number of Chebyshev roots to generate.

    Returns:
    numpy.ndarray: An array of N+1 Chebyshev roots within the interval [x_left, x_right].
    """

    radius = (x_right - x_left) / 2.
    center = (x_right + x_left) / 2.    
    return center + radius * np.cos(np.pi - np.arange(N+1)*np.pi/N)

def interpolateAchevyshev(grade=13):
    """
    Interpolates the function f_a(x) using Chebyshev nodes.
    """
    x = chebyshev_roots(-4, 4, grade)

    f_cheb = f_a(x) 
    f_interL = scipy.interpolate.lagrange(x, f_cheb)
    f_interS = scipy.interpolate.CubicSpline(x, f_cheb)
    xx = np.linspace(-4, 4, 300)
    plt.figure(figsize=(8, 6))
    plt.plot(x, f_cheb, label='Chebyshev nodes', marker='o', color='black', linestyle='None')
    plt.plot(xx, f_interL(xx), label='Funcion interpolada con Lagrange', color='g')  
    plt.plot(xx, f_interS(xx), label='Funcion interpolada con Spline', color='purple')
    plt.plot(xx, f_a(xx), label='Función original', color='black', linestyle='--')
    plt.title('Función interpolada con Chebyshev')
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


def error_b_cheb(nodes_range = range(2, 21)):
    errors = []
    for n in nodes_range:
        x1 = chebyshev_roots(-1, 1, n)
        x2 = chebyshev_roots(-1, 1, n)
        # Create meshgrid
        x1, x2 = np.meshgrid(x1, x2)
        # Generate interpolated values
        grid_z = scipy.interpolate.griddata((x1.flatten(), x2.flatten()), f_b(x1, x2).flatten(), (x1, x2), method='linear')
        # Calculate error
        error = np.abs(f_b(x1, x2) - grid_z)
        
        # Append error to list
        errors.append(np.mean(error))
    x = np.arange(len(nodes_range))
    plt.figure(figsize=(8, 6))
    plt.bar(x, errors, width=0.4, label='Error', color='purple')
    plt.xlabel('Node Quantity')
    plt.ylabel('Error')
    plt.title('Error vs Node Quantity')
    plt.xticks(x, nodes_range)
    plt.legend()
    plt.show()


def comparacion_errores_chebL():
    grade = range(2, 14)
    errors_cheb = []
    errors_no_cheb = []
    
    for n in grade:
        points_cheb = chebyshev_roots(-4, 4, n)
        points_no_cheb = divide_interval(np.linspace(-4, 4, 100), n)
        
        g_cheb = scipy.interpolate.lagrange(points_cheb, f_a(points_cheb))
        g_no_cheb = scipy.interpolate.lagrange(points_no_cheb, f_a(points_no_cheb))
        
        error_cheb = calculate_error(f_a, g_cheb, np.linspace(-4, 4, 100))
        error_no_cheb = calculate_error(f_a, g_no_cheb, np.linspace(-4, 4, 100))
        
        errors_cheb.append(error_cheb)
        errors_no_cheb.append(error_no_cheb)
    
    x = np.arange(len(grade))
    plt.figure(figsize=(8, 6))
    plt.bar(x, errors_cheb, width=0.4, label='Error con Chebyshev ', color='g')
    plt.bar(x + 0.4, errors_no_cheb, width=0.4, label='Error sin Chebyshev ', color='purple')
    plt.xlabel('Grado')
    plt.ylabel('Error')
    plt.title('Comparacioon errores Lagrange (Chebyshev vs sin Chebyshev)')
    plt.xticks(x, grade)
    plt.legend()
    plt.show()

def comparacion_errores_chebS():
    grade = range(2, 14)
    errors_cheb = []
    errors_no_cheb = []
    
    for n in grade:
        points_cheb = chebyshev_roots(-4, 4, n)
        points_no_cheb = divide_interval(np.linspace(-4, 4, 100), n)
        
        g_cheb = scipy.interpolate.CubicSpline(points_cheb, f_a(points_cheb))
        g_no_cheb = scipy.interpolate.CubicSpline(points_no_cheb, f_a(points_no_cheb))
        
        error_cheb = calculate_error(f_a, g_cheb, np.linspace(-4, 4, 100))
        error_no_cheb = calculate_error(f_a, g_no_cheb, np.linspace(-4, 4, 100))
        
        errors_cheb.append(error_cheb)
        errors_no_cheb.append(error_no_cheb)
    
    x = np.arange(len(grade))
    plt.figure(figsize=(8, 6))
    plt.bar(x, errors_cheb, width=0.4, label='Error con Chebyshev ', color='g')
    plt.bar(x + 0.4, errors_no_cheb, width=0.4, label='Error sin Chebyshev ', color='purple')
    plt.xlabel('Grado')
    plt.ylabel('Error')
    plt.title('Comparacion errores Splines (Chebyshev vs sin Chebyshev)')
    plt.xticks(x, grade)
    plt.legend()
    plt.show()

def divide_interval(interval: np.ndarray, n: int) -> np.ndarray:
    points = np.linspace(interval[0], interval[-1], num=n)
    return points

def calculate_error(f, g, interval):
    return scipy.integrate.quad(lambda x: abs((f(x) - g(x))), interval[0], interval[-1])[0]

def main():
    # plot_a()
    #lagrange([5, 8]) #YA GUARDE GRAFICO
    #splines([5, 12]) #YA GUARDE GRAFICO puntos 
    lagrange_splines_equi(8) #YA GUARDE GRAFICO

    # errores_absolutos_equis(8) 
    # errores_absolutos_chebyshev(8)
    # calculate_errors()
    # interpolateAchevyshev()

    # calculate_errors_cheb()
    # comparacion_errores_chebL()
    # comparacion_errores_chebS()
    

    #plot_b()
    #interpolated_b()
    #error_b_cheb()
    #interpolateBchevychev()


    


    
if __name__ == "__main__":
    main()