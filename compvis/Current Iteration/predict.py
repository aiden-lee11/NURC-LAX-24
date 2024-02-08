import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

# Code implentation taken from Vincent's triloc predict.py
# https://github.com/v-rim/triloc/blob/main/triloc/predict.py
class RLS:
    """Implementation from https://github.com/craig-m-k/Recursive-least-squares"""

    def __init__(self, num_vars, lam, delta=1):
        """
        num_vars: number of variables including constant
        lam: forgetting factor, usually very close to 1.
        """
        self.num_vars = num_vars

        # delta controls the initial state.
        self.A = delta * np.matrix(np.identity(self.num_vars))
        self.w = np.matrix(np.zeros(self.num_vars))
        self.w = self.w.reshape(self.w.shape[1], 1)

        # Variables needed for add_obs
        self.lam_inv = lam ** (-1)
        self.sqrt_lam_inv = np.sqrt(self.lam_inv)

        # A priori error
        self.a_priori_error = 0

        # Count of number of observations added
        self.num_obs = 0

    def add_obs(self, x, t):
        """
        Add the observation x with label t.
        x is a column vector as a numpy matrix
        t is a real scalar
        """
        z = self.lam_inv * self.A * x
        alpha = (1 + (x.T @ z).item()) ** (-1)
        self.a_priori_error = t - (self.w.T @ x).item()
        self.w = self.w + (t - alpha * (x.T @ (self.w + t * z)).item()) * z
        self.A -= alpha * z * z.T
        self.num_obs += 1

    def fit(self, X, y):
        """
        Fit a model to X, y.
        X and y are numpy arrays.
        Individual observations in X should have a prepended 1 for the constant coefficient.
        """
        for i, point in enumerate(X):
            x = np.transpose(np.matrix(point))
            self.add_obs(x, y[i])

    def get_error(self):
        """
        Finds the a priori (instantaneous) error.
        Does not calculate the cumulative effect
        of round-off errors.
        """
        return self.a_priori_error

    def predict(self, x):
        """
        Predict the value of observation x. x should be a numpy matrix (col vector)
        """
        return float(self.w.T * x)



class RecursivePolynomialFit:
    def __init__(self, degree, forgetting_factor=1):
        self.degree = degree
        self.forgetting_factor = forgetting_factor

        self.RLS = RLS(degree + 1, forgetting_factor)

    def add_point(self, x, y):
        x_pow = [x**i for i in range(self.degree + 1)]
        x_pow = np.array(x_pow)[:, None]

        self.RLS.add_obs(x_pow, y)

    def get_coef(self):
        return np.array(self.RLS.w.T).flatten()

    def solve(self, y):
        coef = np.array(self.RLS.w.T).flatten()
        coef[0] -= y
        return Polynomial(coef).roots()
    
    def plug_in(self, inter_time):
        coef = np.array(self.RLS.w.T).flatten()
        return sum([c * inter_time**i for i, c in enumerate(coef)])
        


    def reset(self):
        self.RLS = RLS(self.degree, self.forgetting_factor)


def quadratic_regression_test():
    # Predicting a quadratic function
    test_size = 1000

    # Test function
    def f(x):
        return 0.2 * x**2 - 3.8 * x - 5.1

    # Gaussian noise to be added to the quadratic signal
    noise = np.random.randn(test_size)
    # You can play around with other noise (like sinusoidal)
    # noise = [np.sin(2*np.pi*i/13) for i in range(test_size)]
    y = np.array([f(i) for i in range(test_size)])
    noisy_y = y + noise
    lam = 0.98
    LS = RLS(3, lam, 1)
    # Not using the RLS.fit function because I want to remember all the predicted values
    pred_x = []
    pred_y = []
    for i in range(test_size):
        x = np.matrix(np.zeros((1, 3)))
        x[0, 0] = 1
        x[0, 1] = i
        x[0, 2] = i**2
        pred_x.append(i)
        pred_y.append((x @ LS.w).item())

        LS.add_obs(x.T, noisy_y[i])

    print("Predicted equation in the form Ax^2 + Bx + C")
    print(f"A = {LS.w[2, 0]} | Real value = 0.2")
    print(f"B = {LS.w[1, 0]} | Real value = -3.8")
    print(f"C = {LS.w[0, 0]} | Real value = -5.1")
    # plot the predicted values against the non-noisy output
    plt.plot(pred_x, y - pred_y)
    plt.title("Error as more points are added")
    plt.show()


def RPF_test():
    coef = np.array([1, 2, 3])
    target_y = 6
    eq = coef.copy()
    eq[0] -= target_y

    def f(x):
        return sum([c * x**i for i, c in enumerate(coef)])
    X = range(-100, 100)
    points = np.array([(i, f(i)) for i in X])
    
    rpf = RecursivePolynomialFit(2)
    for x, y in points:
        rpf.add_point(x, y)

    print(f"Expected coefficients = {coef}")
    print(f"{rpf.get_coef().round(3) = }")
    print()

    print(f"Solving for f(x)={target_y}")
    print(f"Expected solutions = {Polynomial(eq).roots()}")
    print(f"{rpf.solve(target_y).round(3) = }")
    print()

    rpf.reset()
    print("Reset RPF")
    print(f"{rpf.get_coef().round(3) = }")


if __name__ == "__main__":
    # quadratic_regression_test()
    RPF_test()

    # NOTE: To deal with the output being dimensions in R^3, it seems sufficient
    # to deal with each dimension separately.
    # See https://math.stackexchange.com/q/2688132

"""
To solve for time, solve the coeff of y polynomial = 0 (y-plane under the cams). This solves for the time of intersection
between the ball and end-defector plane. After that, use the solution to that equation as the value of t to plug into 
x and z equations to predict where the end-defector should move to.
"""