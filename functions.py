import numpy as np

# DomFa = [-4, 4]
def f_a(x): 
    return 0.3 ** np.linalg.norm(x) * np.sin(4 * x) - np.tanh(2 * x) + 2

# DomFb = [-1, 1]
def f_b(x1, x2):
    return (
        0.75 * np.exp(-((10 * x1 - 2) ** 2 + (9 * x2 - 2) ** 2) / 4)
        + 0.65 * np.exp(-((9 * x1 + 1) ** 2 / 9 + (10 * x2 + 1) ** 2 / 2))
        + 0.55 * np.exp(-((9 * x1 - 6) ** 2 + (9 * x2 - 3) ** 2) / 4)
        - 0.01 * np.exp(-((9 * x1 - 7) ** 2 + (9 * x2 - 3) ** 2) / 4)
    )

