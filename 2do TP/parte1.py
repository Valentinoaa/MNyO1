
import numpy as np
import matplotlib.pyplot as plt

def exponential_growth(N0, r, t):
    return N0 * np.exp(r * t)

def logistic_growth(N0, r, K, t):
    return (K * N0 * np.exp(r * t)) / (K + N0 * (np.exp(r * t) - 1))

def plotNvsT():
    # Time range and carrying capacity K for logistic growth
    t = np.linspace(0, 10, 100)
    K = 100  # Carrying capacity

    # Scenarios with different r and N0
    scenarios = [
        {'N0': 10, 'r': 0.1, 'label': 'N0=10, r=0.1'},
        {'N0': 100, 'r': 0.1, 'label': 'N0=100, r=0.1'},
        {'N0': 10, 'r': 0.5, 'label': 'N0=10, r=0.5'},
        {'N0': 100, 'r': 0.5, 'label': 'N0=100, r=0.5'}
    ]

    # Plotting
    plt.figure(figsize=(14, 8))

    for scenario in scenarios:
        exp_growth = exponential_growth(scenario['N0'], scenario['r'], t)
        log_growth = logistic_growth(scenario['N0'], scenario['r'], K, t)
        
        plt.plot(t, exp_growth, '--', label=f'Exponential {scenario["label"]}')
        plt.plot(t, log_growth, '-', label=f'Logistic {scenario["label"]}')

    plt.title('Impact of Growth Rate and Initial Population on Population Dynamics')
    plt.xlabel('Time (t)')
    plt.ylabel('Population Size (N(t))')
    plt.legend()
    plt.grid(True)
    plt.show()





def f_exponencial(n, t, r):
    return r * n

def f_logistic(n, t, r, K=100):
    return r * n * (1 - n / K)

def euler_method(f, n0, t0, t_end, h, r, original_function):
    """
    Euler's method for solving ordinary differential equations (ODEs).
    
    Parameters:
    - f: The function representing the ODE dy/dt = f(n, t).
    - n0: The initial value of the dependent variable.
    - t0: The initial value of the independent variable.
    - t_end: The end value of the independent variable.
    - h: The step size.
    
    Returns:
    - n_values: A list of the dependent variable values at each time step.
    - t_values: A list of the corresponding time values.
    """
    
    t = t0
    n = n0
    n_values = []
    t_values = []
    relative_error = []
    
    while t < t_end:
        n_values.append(n)
        t_values.append(t)
        relative_error.append(abs(n - original_function(1, r, t)) / original_function(1, r, t))
        n += h * f(n, t, r)
        t += h
        
    return n_values, t_values, relative_error

def runge_kutta_4(f, n0, t0, t_end, h, r, original_function):
    """
    Fourth-order Runge-Kutta method for solving ordinary differential equations (ODEs).
    
    Parameters:
    - f: The function representing the ODE dy/dt = f(n, t).
    - n0: The initial value of the dependent variable.
    - t0: The initial value of the independent variable.
    - t_end: The end value of the independent variable.
    - h: The step size.
    
    Returns:
    - n_values: A list of the dependent variable values at each time step.
    - t_values: A list of the corresponding time values.
    """
    
    t = t0
    n = n0
    n_values = []
    t_values = []
    relative_error = []
    
    while t < t_end:
        n_values.append(n)
        t_values.append(t)
        relative_error.append(abs(n - original_function(1, r, t)) / original_function(1, r, t))
        k1 = h * f(n, t, r)
        k2 = h * f(n + 0.5 * k1, t + 0.5 * h, r)
        k3 = h * f(n + 0.5 * k2, t + 0.5 * h, r)
        k4 = h * f(n + k3, t + h, r)
        
        n += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
        
    return n_values, t_values, relative_error

def logistic_growth100(N0, r, t):
    return logistic_growth(N0, r, 100, t)

# Ejemplos de uso de las funciones definidas anteriormente
def plotEulerVsRK4():
    # Ejemplo de uso de Euler Method y Runge-Kutta 4th Order para función exponencial con r > 0 y r < 0
    pos_exp_euler_n, pos_exp_euler_t, pos_exp_euler_error = euler_method(f_exponencial, 1, 0, 100, 0.1, 0.1, exponential_growth)
    pos_exp_rk4_n, pos_exp_rk4_t, pos_exp_rk4_error = runge_kutta_4(f_exponencial, 1, 0, 100, 0.1, 0.1, exponential_growth)
    neg_exp_euler_n, neg_exp_euler_t, neg_exp_euler_error = euler_method(f_exponencial, 1, 0, 100, 0.1, -0.1, exponential_growth)
    neg_exp_rk4_n, neg_exp_rk4_t, neg_exp_rk4_error = runge_kutta_4(f_exponencial, 1, 0, 100, 0.1, -0.1, exponential_growth)

    # Ejemplo de uso de Euler Method para función logística con r > 0 y r < 0
    pos_log_euler_n, pos_log_euler_t, pos_log_euler_error = euler_method(f_logistic, 1, 0, 100, 0.1, 0.1, logistic_growth100)
    neg_log_euler_n, neg_log_euler_t, neg_log_euler_error = euler_method(f_logistic, 1, 0, 100, 0.1, -0.1, logistic_growth100)

    # Ejemplo de uso de Runge-Kutta 4th Order para función logística con r > 0 y r < 0
    pos_log_rk4_n, pos_log_rk4_t, pos_log_rk4_error = runge_kutta_4(f_logistic, 1, 0, 100, 0.1, 0.1, logistic_growth100)
    neg_log_rk4_n, neg_log_rk4_t, neg_log_rk4_error = runge_kutta_4(f_logistic, 1, 0, 100, 0.1, -0.1, logistic_growth100)

    # Plot de los resultados
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(neg_log_euler_t, neg_log_euler_error, label='Euler Method (r < 0)')
    plt.plot(pos_log_euler_t, pos_log_euler_error, label='Euler Method (r > 0)')
    plt.title('Euler log')
    plt.xlabel('Time (t)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(neg_log_rk4_t, neg_log_rk4_error, label='Runge-Kutta 4th Order (r < 0)')
    plt.plot(pos_log_rk4_t, pos_log_rk4_error, label='Runge-Kutta 4th Order (r > 0)')
    plt.title('RK4 log')
    plt.xlabel('Time (t)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(pos_exp_euler_t, pos_exp_euler_error, label='Euler Method (r > 0)')
    plt.plot(neg_exp_euler_t, neg_exp_euler_error, label='Euler Method (r < 0)')
    plt.title('Exp euler')
    plt.xlabel('Time (t)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(neg_exp_rk4_t, neg_exp_rk4_error, label='Runge-Kutta 4th Order (r < 0)')
    plt.plot(pos_exp_rk4_t, pos_exp_rk4_error, label='Runge-Kutta 4th Order (r > 0)')
    plt.title('exp log')
    plt.xlabel('Time (t)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.show()


def exponential_growth_derivative(N0, r, t):
    return r * N0 * np.exp(r * t)

def logistic_growth_derivative(N0, r, K, t):
    return (r * K * N0 * np.exp(r * t)) / (K + N0 * (np.exp(r * t) - 1)) - (K * r  * np.exp(2*r * t) * N0 ** 2) / (K + N0 * (np.exp(r * t) - 1))**2


# Time range for visualization
t = np.linspace(0, 100, 100)

# Parameters for scenarios
r_positive = 0.5
r_negative = -0.5
N0 = 10
K = 100

# Calculate N and N' for positive r
N_exp_positive = exponential_growth(N0, r_positive, t)
N_prime_exp_positive = exponential_growth_derivative(N0, r_positive, t)

N_log_positive = logistic_growth(N0, r_positive, K, t)
N_prime_log_positive = logistic_growth_derivative(N0, r_positive, K, t)

# Calculate N and N' for negative r
N_exp_negative = exponential_growth(N0, r_negative, t)
N_prime_exp_negative = exponential_growth_derivative(N0, r_negative, t)

N_log_negative = logistic_growth(N0, r_negative, K, t)
N_prime_log_negative = logistic_growth_derivative(N0, r_negative, K, t)

def plotNvsNPrime():
    # Plotting N vs N'
    plt.figure(figsize=(14, 7))

    # Plot for positive r
    plt.subplot(1, 2, 1)
    #plt.plot(N_exp_positive, N_prime_exp_positive, 'g-', label=f'Función exponencial')
    plt.plot(N_log_positive, N_prime_log_positive, 'b-', label=f'Función logística')
    plt.title('N vs. N\' con r positivo(r = 0.5)')
    plt.xlabel('Tamaño de la población N(t)')
    plt.ylabel("Rate of Change of Population N'(t)")
    plt.grid(True)
    plt.legend()

    # Plot for negative r
    plt.subplot(1, 2, 2)
    plt.plot(N_exp_negative, N_prime_exp_negative, 'r-', label=f'Función exponencial')
    plt.plot(N_log_negative, N_prime_log_negative, 'm-', label=f'Función logística')
    plt.title('N vs. N\' con r negativo(r = -0.5)')
    plt.xlabel('Tamaño de la población N(t)')
    plt.ylabel("Rate of Change of Population N'(t)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plotVectorField():
    # Parameters for the vector field
    r = 0.1  # Growth rate
    N_max = 20  # Maximum population size for the plot
    t_max = 10  # Maximum time for the plot

    # Create a grid of t and N values
    t, N = np.meshgrid(np.linspace(0, t_max, 20), np.linspace(0, N_max, 20))

    # The change in N and t (dt is always 1 for visualization, dN/dt = rN)
    dt = np.ones_like(t)
    dN = r * N

    # Normalize vectors for uniform appearance
    norm = np.sqrt(dt**2 + dN**2)
    dt /= norm
    dN /= norm

    # Creating the vector field plot

    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.quiver(t, N, dt, dN, color='blue')
    plt.xlabel('Time ($t$)')
    plt.ylabel('Population Size ($N$)')
    plt.title('Vector Field of Exponential Growth ($\\frac{dN}{dt} = rN$, $r > 0$)')
    plt.xlim([0, t_max])
    plt.ylim([0, N_max])
    plt.grid(True)

    # Adjust the parameter for decay
    r_decay = -0.1  # Negative growth rate for decay

    # The change in N with r < 0
    dN_decay = r_decay * N

    # Normalize vectors for uniform appearance in the decay scenario
    norm_decay = np.sqrt(dt**2 + dN_decay**2)
    dt_decay = dt / norm_decay
    dN_decay /= norm_decay

    # Creating the vector field plot for decay
    plt.subplot(1, 2, 2)
    plt.quiver(t, N, dt_decay, dN_decay, color='red')
    plt.xlabel('Time ($t$)')
    plt.ylabel('Population Size ($N$)')
    plt.title('Vector Field of Exponential Decay ($\\frac{dN}{dt} = rN$, $r < 0$)')
    plt.xlim([0, t_max])
    plt.ylim([0, N_max])
    plt.grid(True)
    plt.show()

    plt.show()


# Parameters for the logistic growth vector field
def plotLogisticVectorField():
    r_logistic = 0.1  # Growth rate
    K_logistic = 130   # Carrying capacity
    t_max = 100         # Maximum time for the plot

    # Create a grid of t and N values for logistic growth
    t_logistic, N_logistic = np.meshgrid(np.linspace(0, t_max, 20), np.linspace(0, 140, 20))

    # The change in N according to the logistic model
    dN_logistic = r_logistic * N_logistic * (1 - N_logistic / K_logistic)
    dt_logistic = np.ones_like(t_logistic)

    # Normalize vectors for uniform appearance in the logistic scenario
    norm_logistic = np.sqrt(dt_logistic**2 + dN_logistic**2)
    dt_logistic /= norm_logistic
    dN_logistic /= norm_logistic

    # Creating the vector field plot for logistic growth
    plt.figure(figsize=(8, 6))
    plt.quiver(t_logistic, N_logistic, dt_logistic, dN_logistic, color='green')
    plt.xlabel('Time ($t$)')
    plt.ylabel('Population Size ($N$)')
    plt.title('Vector Field of Logistic Growth ($\\frac{dN}{dt} = rN(1 - \\frac{N}{K})$)')
    plt.xlim([0, t_max])
    plt.ylim([0, 140])
    plt.grid(True)
    plt.show()



def plotRateOfChange():
    t = np.linspace(0, 100, 100)
    K = 100  # Carrying capacity

    plt.figure(figsize=(14, 8))

    plt.subplot(1, 2, 1)
    plt.plot(t, N_prime_exp_positive, label='N0=10, r=0.5')
    plt.plot(t, N_prime_exp_negative, label='N0=10, r=-0.5')
    plt.title('Exponential Growth Rate')
    plt.xlabel('Time (t)')
    plt.ylabel('Rate of Change of Population N(t)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, N_prime_log_positive, label='N0=10, r=0.5, K=100')
    plt.plot(t, N_prime_log_negative, label='N0=10, r=-0.5, K=100')
    plt.title('Logistic Growth Rate')
    plt.xlabel('Time (t)')
    plt.ylabel('Rate of Change of Population N(t)')
    plt.legend()

    plt.show()

def main():
    plotNvsT()
    plotEulerVsRK4()
    plotNvsNPrime()
    plotVectorField()
    plotLogisticVectorField()
    plotRateOfChange()

if __name__ == '__main__':
    main()
