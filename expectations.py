import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fmin, minimize
from scipy.stats import norm
import torch


########### Linear expectations ##############

class binomial_model:
    """
    Linear expectation via symmetric binomial model with standard deviation sigma.

    Attributes:
        vol (float): standard deviation.
        model (str): model type.
    
    Methods:
        __init__(self, sigma)
            Initializes the binomial_model object with given volatility.

        expect(self, f)
            Computes the expectation E[f(zeta)], where zeta is distributed according to the binomial model.
        
        summary(self, log_file)
            Writes a summary of the model information to a log file.
    """

    def __init__(self, sigma) -> None:
        self.vol = sigma
        self.model = 'Binomial'

    def expect(self, f: callable) -> float:
        return np.divide(f(self.vol) + f(- self.vol), 2)

    def error_prop(self):
        return self.vol
    
    def summary(self, log_file) -> None:
        log_file.write("\n" + f"1-d {self.model}. Std: {self.vol}")

    

class trinomial_model:
    """
    Linear expectation via symmetric trinomial model with standard deviation sigma.

    Attributes:
        vol (float): standard deviation.
        scale (float): upper and lower increments.
        model (str): model type.
    
    Methods:
        __init__(self, sigma)
            Initializes the trinomial_model object with given volatility.

        expect(self, f)
            Computes the expectation E[f(zeta)], where zeta is distributed according to the trinomial model.
        
        summary(self, log_file)
            Writes a summary of the model information to a log file.
    """

    def __init__(self, sigma) -> None:
        self.vol = sigma
        self.scale = sigma * np.sqrt(1.5)
        self.model = 'Trinomial'

    def expect(self, f: callable) -> float:
        res = np.divide(f(self.scale) + f(0) + f(-self.scale), 3.)
        return res
    
    def error_prop(self):
        return self.scale
    
    def summary(self, log_file) -> None:
        log_file.write("\n" + f"1-d {self.model}. Std: {self.vol}")



class normal_model:
    """
    Linear expectation via normal model with standard deviation sigma.

    Attributes:
        vol (float): standard deviation.
        quad_ord (int): order of the gaussian quadrature algorithm used to compute the expectation.
        model (str): model type.

    Methods:
        __init__(self, sigma, quad_ord)
            Initializes the normal_model object with given volatility and quadrature order for the numerical integration.

        expect(self, f)
            Computes the expectation E[f(zeta)], where zeta is distributed according to the normal model.

        summary(self, log_file)
            Writes a summary of the model information to a log file.
    """

    def __init__(self, sigma, quad_ord = 16) -> None:
        self.vol = sigma
        self.quad_ord = quad_ord
        self.model = "Normal"

    def expect(self, f: callable) -> float:
        return integrate.fixed_quad(lambda s: f(s) * norm.pdf(s, loc=0, scale=self.vol), -5 * self.vol, 5 * self.vol, n=self.quad_ord)[0]
    
    def error_prop(self):
        return 1.65 * self.vol
    
    def summary(self, log_file) -> None:
        log_file.write("\n" + f"1-d {self.model}. Std: {self.vol}")



class uniform_model:
    """
    Linear expectation via symmetric uniform model with standard deviation sigma.

    Attributes:
        vol (float): standard deviation.
        scale (float): 0.5 * interval width
        quad_ord (int): order of the gaussian quadrature algorithm used to compute the expectation.
        model (str): model type.

    Methods:
        __init__(self, sigma, quad_ord)
            Initializes the uniform_model object with given volatility and quadrature order for the numerical integration.

        expect(self, f)
            Computes the expectation E[f(zeta)], where zeta is distributed according to the uniform model.

        summary(self, log_file)
            Writes a summary of the model information to a log file.
    """

    def __init__(self, sigma, quad_ord = 16) -> None:
        self.vol = sigma
        self.scale = np.sqrt(3) * sigma
        self.quad_ord = quad_ord
        self.model = "Uniform"

    def expect(self, f: callable) -> float:
        return integrate.fixed_quad(f, -self.scale, self.scale, n=self.quad_ord)[0] / (2. * self.scale)
    
    def error_prop(self):
        return self.scale
    
    def summary(self, log_file) -> None:
        log_file.write("\n" + f"1-d {self.model}. Std: {self.vol}")




########### sublinear expectations ##############

class sub_binomial_model:
    """
    Sublinear expectation corresponding to a symmetric binomial model with uncertain standard deviation sigma in [sigma_l, sigma_u].

    Attributes:
        vol_l (float): Lower bound of volatility.
        vol_u (float): Upper bound of volatility.
        vol0 (float): Initial guess for volatility, calculated as the average of vol_l and vol_u.
        vol_trained (bool): Flag indicating whether the optimal volatility has been filled.
        vol_opt (np.ndarray): Array containing the optimal volatility value.
        model (str): Description of the model.

    Methods:
        __init__(self, sigma_l, sigma_u)
            Initializes the sub_binomial_model object with given volatility bounds.
        
        expect(self, f)
            Computes the expectation of a given function with respect to the uncertain binomial model.
        
        fill_vol(self, vol)
            Fills in the optimal volatility value for the model.
        
        summary(self, log_file)
            Writes a summary of the model information to a log file.

    Example:
        model = sub_binomial_model(0.2, 0.5)
        # Estimate the expectation of a function using the uncertain volatility.
        expectation, optimal_volatility = model.expect(lambda x: x**2)
        # Fill in the optimal volatility value.
        model.fill_vol(optimal_volatility)
        # Summarize the model information in a log file.
        with open('model_summary.txt', 'a') as log_file:
            model.summary(log_file)
    """

    def __init__(self, sigma_l, sigma_u, opt_method='Nelder-Mead') -> None:
        self.vol_l = sigma_l
        self.vol_u = sigma_u
        self.vol0 = (sigma_l + sigma_u) / 2.
        self.vol_trained = False
        self.vol_opt = np.array([])
        self.opt_method = opt_method
        self.model = f'1-d binomial model with volatility uncertainty. Volatility bounds: {self.vol_l, self.vol_u}.'

    def expect(self, f: callable) -> float:
        if self.vol_trained:
            return np.divide(f(self.vol_opt) + f(- self.vol_opt), 2), self.vol_opt
        else:
            expectation = minimize(lambda vol: - np.divide(f(vol) + f(- vol), 2), x0=self.vol0, method=self.opt_method, bounds=[(self.vol_l, self.vol_u)])
            return - expectation.fun, expectation.x[0]
        
    def fill_vol(self, vol):
        self.vol_opt = vol
        self.vol_trained = True
    
    def summary(self, log_file) -> None:
        log_file.write("\n" + self.model)




########### risk measures ##############

class RhoTheta:

    def __init__(self, alpha, m, t) -> None:
        self.alpha = alpha
        self.m = m
        self.t = t

    def evaluate(self, theta, expectation, f, x):
        integrand = lambda zeta: np.exp(self.alpha * f(x + self.m * self.t + np.sqrt(self.t) * zeta)
                                                    + self.alpha * theta * np.sqrt(self.t) * zeta)
        expect, vol_opt = expectation.expect(integrand)
        return theta * self.m * self.t + np.log(expect) / self.alpha, vol_opt
    
    def ct_loss(self, theta, expectation):
        res, _ = self.evaluate(theta, expectation, lambda s: 0, 0.)
        return res

    def hedge_loss(self, theta, expectation, f, x):
        res, _ = self.evaluate(theta, expectation, f, x)
        return res
    


class RhoThetaCertain:

    def __init__(self, alpha, m, t, bid=False) -> None:
        self.alpha = alpha
        self.m = m
        self.t = t
        self.mult = 1.
        if bid:
            self.mult = -1.

    def evaluate(self, theta, expectation, f, x):
        integrand = lambda zeta: self.mult * np.exp(self.alpha * self.mult * f(x + self.m * self.t + np.sqrt(self.t) * zeta)
                                                    + self.alpha * theta * np.sqrt(self.t) * zeta)
        expect = expectation.expect(integrand)
        return theta * self.m * self.t + np.log(self.mult * expect) / self.alpha
    
    def ct_loss(self, theta, expectation):
        res = self.evaluate(theta, expectation, lambda s: 0, 0.)
        return res

    def hedge_loss(self, theta, expectation, f, x):
        res = self.evaluate(theta, expectation, f, x)
        return res