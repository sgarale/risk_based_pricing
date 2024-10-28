import math
import numpy as np
from scipy.optimize import minimize

def certain_iterations(x_levels, x_fine, option, step_nr, rho_theta, make_expect, bid=False):
    """
    Performs the iterations of the one-step pricing functional computing the optimal hedging strategy at each step. It works with linear expectations, hence without uncertainty.

    Parameters:
    ----------
    x_levels : array-like
        Discrete levels of the underlying for which the hedging strategy is computed.
    x_fine : array-like
        Fine-grained points for interpolating the hedging strategy.
    option : object
        EuropeanOption object with a `payoff` method, defining the payoff function.
    step_nr : int
        Number of steps in the time discretization.
    rho_theta : object
        An entropic risk measure providing `ct_loss` and `hedge_loss` methods to evaluate loss functions for the optimal strategy.
    make_expect : callable
        Function that returns an expectation object with an `expect` method to compute expected values.
    bid : bool, optional
        If True, performs the iterations for the bid price. Default is False.

    Returns:
    -------
    I_t_dict : dict
        Dictionary with keys as iteration steps and values as the computed `I_t` values at each step.
    theta_dict : dict
        Dictionary with keys as iteration steps and values as the computed optimal `theta` values at each step. The key 0 corresponds to the optimal theta for the renormalization term.
    expect_dict : dict
        Dictionary with keys as iteration steps and values as the expectation objects used in each iteration.
    """
    mult = 1.
    if bid:
        mult = -1.
    
    # computing c(t)
    expectation = make_expect()

    theta_opt = minimize(rho_theta.ct_loss, x0=0., args=(expectation))
    c_t = theta_opt.fun
    theta0_star = theta_opt.x[0]

    # initializing the dictionary of the iterations
    I_t_dict = {0: c_t}
    theta_dict = {0: theta0_star}
    expect_dict = {0: expectation}

    theta_star_list = []

    # computing the optimal hedging strategy for a set of points
    iteration = 1
    expectation = make_expect()
    for x in x_levels:
        theta_opt = minimize(rho_theta.hedge_loss, x0=0., args=(expectation, option.payoff, x))
        theta_star = theta_opt.x[0]
        theta_star_list.append(theta_star)

    # interpolating the hedging strategy on the points x_fine
    theta_star_fine = np.interp(x_fine, x_levels, theta_star_list)

    # computing I(t)f from the interpolation
    if expectation.model in ["Normal", "Uniform"]: # the model that use numerical integration do not allow for vectorial evaluation
        I_t_fine = []
        for idx, theta in enumerate(theta_star_fine):
            I_t_fine.append(mult * (rho_theta.evaluate(theta, expectation, option.payoff, x_fine[idx]) - c_t))
        I_t_fine = np.array(I_t_fine)
    else:
        I_t_fine = mult * (rho_theta.evaluate(theta_star_fine, expectation, option.payoff, x_fine) - c_t)

    # saving the output of the first step
    expect_dict[iteration] = expectation
    theta_dict[iteration] = theta_star_list
    I_t_dict[iteration] = I_t_fine


    while iteration < step_nr:
        iteration += 1
        expectation = make_expect()
        
        def f_i(x):
            return np.interp(x, x_fine, I_t_fine)

        theta_star_list = []

        for x in x_levels:
            theta_opt = minimize(rho_theta.hedge_loss, x0=0., args=(expectation, f_i, x))
            theta_star = theta_opt.x[0]
            theta_star_list.append(theta_star)

        # interpolating the hedging strategy on the points x_fine
        theta_star_fine = np.interp(x_fine, x_levels, theta_star_list)

        # computing I(t)f from the interpolation
        if expectation.model in ["Normal", "Uniform"]: # the model that use numerical integration do not allow for vectorial evaluation
            I_t_fine_new = []
            for idx, theta in enumerate(theta_star_fine):
                I_t_fine_new.append(mult * (rho_theta.evaluate(theta, expectation, f_i, x_fine[idx]) - c_t))
            I_t_fine = np.array(I_t_fine_new)
        else:
            I_t_fine = mult * (rho_theta.evaluate(theta_star_fine, expectation, f_i, x_fine) - c_t)

        # saving the output of the first step
        expect_dict[iteration] = expectation
        theta_dict[iteration] = theta_star_list
        I_t_dict[iteration] = I_t_fine

    return I_t_dict, theta_dict, expect_dict



def unc_iterations(x_levels, x_fine, option, step_nr, rho_theta, make_expect, bid=False):
    """
    Performs the iterations of the one-step pricing functional computing the optimal hedging strategy at each step. It works with sublinear expectations, hence with uncertainty.

    Parameters:
    ----------
    x_levels : array-like
        Discrete levels of the underlying for which the hedging strategy is computed.
    x_fine : array-like
        Fine-grained points for interpolating the hedging strategy.
    option : object
        EuropeanOption object with a `payoff` method, defining the payoff function.
    step_nr : int
        Number of steps in the time discretization.
    rho_theta : object
        An entropic risk measure providing `ct_loss` and `hedge_loss` methods to evaluate loss functions for the optimal strategy.
    make_expect : callable
        Function that returns an expectation object with an `expect` method to compute expected values.
    bid : bool, optional
        If True, performs the iterations for the bid price. Default is False.

    Returns:
    -------
    I_t_dict : dict
        Dictionary with keys as iteration steps and values as the computed `I_t` values at each step.
    theta_dict : dict
        Dictionary with keys as iteration steps and values as the computed optimal `theta` values at each step. The key 0 corresponds to the optimal theta for the renormalization term.
    expect_dict : dict
        Dictionary with keys as iteration steps and values as the expectation objects used in each iteration.
    """
    mult = 1.
    if bid:
        mult = -1.

    # computing c(t)
    expectation = make_expect()

    theta_opt = minimize(rho_theta.ct_loss, x0=0., args=(expectation))
    c_t = theta_opt.fun
    theta0_star = theta_opt.x[0]

    vol_opt = rho_theta.evaluate(theta0_star, expectation, lambda s: 0, 0.)[1]
    expectation.fill_vol(np.array([vol_opt]))

    # initializing the dictionary of the iterations
    I_t_dict = {0: c_t}
    theta_dict = {0: theta0_star}
    expect_dict = {0: expectation}

    theta_star_list = []

    # computing the optimal hedging strategy for a set of points
    iteration = 1
    expectation = make_expect()
    for x in x_levels:
        theta_opt = minimize(rho_theta.hedge_loss, x0=0., args=(expectation, lambda s: mult * option.payoff(s), x))
        theta_star = theta_opt.x[0]
        theta_star_list.append(theta_star)

    # computing the worst-case volatility at the chosen points
    vol_opt = []
    for i in range(x_levels.shape[0]):
        vol_opt_tmp = rho_theta.evaluate(theta_star_list[i], expectation, lambda s: mult * option.payoff(s), x_levels[i])[1]
        vol_opt.append(vol_opt_tmp)
    # interpolating the volatilities and saving them in the expectation object
    vol_fine = np.interp(x_fine, x_levels, vol_opt)
    expectation.fill_vol(np.array(vol_fine))

    # interpolating the hedging strategy on the points x_fine
    theta_star_fine = np.interp(x_fine, x_levels, theta_star_list)

    # computing I(t)f from the interpolation
    I_t_fine = mult * (rho_theta.evaluate(theta_star_fine, expectation, lambda s: mult * option.payoff(s), x_fine)[0] - c_t)

    # saving the output of the first step
    expect_dict[iteration] = expectation
    theta_dict[iteration] = theta_star_list
    I_t_dict[iteration] = I_t_fine


    while iteration < step_nr:
        iteration += 1
        expectation = make_expect()
        
        def f_i(x):
            return mult * np.interp(x, x_fine, I_t_fine)

        theta_star_list = []

        for x in x_levels:
            theta_opt = minimize(rho_theta.hedge_loss, x0=0., args=(expectation, f_i, x))
            theta_star = theta_opt.x[0]
            theta_star_list.append(theta_star)
        
        # computing the worst-case volatility at the chosen points
        vol_opt = []
        for i in range(x_levels.shape[0]):
            vol_opt_tmp = rho_theta.evaluate(theta_star_list[i], expectation, f_i, x_levels[i])[1]
            vol_opt.append(vol_opt_tmp)
        # interpolating the volatilities and saving them in the expectation object
        vol_fine = np.interp(x_fine, x_levels, vol_opt)
        expectation.fill_vol(np.array(vol_fine))

        # interpolating the hedging strategy on the points x_fine
        theta_star_fine = np.interp(x_fine, x_levels, theta_star_list)

        # computing I(t)f from the interpolation
        I_t_fine = mult * (rho_theta.evaluate(theta_star_fine, expectation, f_i, x_fine)[0] - c_t)

        # saving the output of the first step
        expect_dict[iteration] = expectation
        theta_dict[iteration] = theta_star_list
        I_t_dict[iteration] = I_t_fine

    return I_t_dict, theta_dict, expect_dict



def G_expectation(x_levels, x_fine, option, step_nr, step_size, make_expect, bid=False):
    """
    Chernoff approximation for the G-expectation with generator 0.5 * E[zeta^T D^2f(x) zeta].
    """
    mult = 1.
    if bid:
        mult = -1.

    # initializing the dictionary of the iterations
    J_t_dict = {}
    expect_dict = {}

    theta_star_list = []

    # computing the worst-case volatility for a set of points
    iteration = 1
    expectation = make_expect()
    vol_opt = []
    for x in x_levels:
        _, vol_opt_tmp = expectation.expect(lambda v: mult * option.payoff(x + math.sqrt(step_size) * v))
        vol_opt.append(vol_opt_tmp)

    # interpolating the volatilities and saving them in the expectation object
    vol_fine = np.interp(x_fine, x_levels, vol_opt)
    expectation.fill_vol(np.array(vol_fine))

    # computing J(t)f from the interpolation of the volatilities
    J_t_fine = mult * expectation.expect(lambda v: mult * option.payoff(x_fine + math.sqrt(step_size) * v))[0]

    # saving the output of the first step
    expect_dict[iteration] = expectation
    J_t_dict[iteration] = J_t_fine


    while iteration < step_nr:
        iteration += 1
        expectation = make_expect()
        
        def f_i(x):
            return np.interp(x, x_fine, J_t_fine)

        # computing the worst-case volatility at the chosen points
        vol_opt = []
        for x in x_levels:
            _, vol_opt_tmp = expectation.expect(lambda v: mult * f_i(x + math.sqrt(step_size) * v))
            vol_opt.append(vol_opt_tmp)
        
        # interpolating the volatilities and saving them in the expectation object
        vol_fine = np.interp(x_fine, x_levels, vol_opt)
        expectation.fill_vol(np.array(vol_fine))

        # computing J(t)f from the interpolation of the volatilities
        J_t_fine = mult * expectation.expect(lambda v: mult * f_i(x_fine + math.sqrt(step_size) * v))[0]

        # saving the output of the first step
        expect_dict[iteration] = expectation
        J_t_dict[iteration] = J_t_fine

    return J_t_dict, expect_dict