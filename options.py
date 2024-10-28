import numpy as np
from scipy.optimize import fmin, minimize
from scipy.stats import norm


def black_scholes_call(x, K, T, vol, rate):
    d1 = (np.log(x/K) + (rate + 0.5 * np.power(vol, 2)) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return x * norm.cdf(d1) - K * np.exp(- rate * T) * norm.cdf(d2)


def bachelier_call(x, K, T, vol):
    d = (x - K) / (vol * np.sqrt(T))
    return (x - K) * norm.cdf(d) + vol * np.sqrt(T) * norm.pdf(d)



class EuropeanOption:

    def blackscholes_price(self, x, T, vol, rate=0.):
        raise ValueError("General European Option does not have a blackscholes price")
    
    def bachelier_price(self, x, T, vol, rate):
        raise ValueError("General European Option does not have a blackscholes price")
    
    def bs_implied_vol(self, mkt_price, x, T, rate=0., vol0 = 0.2):
        loss = lambda vol: np.power(self.blackscholes_price(x, T, vol, rate) - mkt_price, 2)
        opt = minimize(loss, x0=[vol0], method='Nelder-Mead', bounds=[(0.00000001, np.inf)])
        return opt.x[0]
    
    def bach_implied_vol(self, mkt_price, x, T, rate=None, vol0=0.2):
        loss = lambda vol: np.power(self.bachelier_price(x, T, vol, rate) - mkt_price, 2)
        opt = fmin(loss, [vol0], disp=False)
        return opt[0]



class BullCallSpread(EuropeanOption):

    def __init__(self, strike_low: float, strike_high: float) -> None:
        super().__init__()
        self.Kl = float(strike_low)
        self.Ku = float(strike_high)

    def payoff(self, x):
        return np.maximum(x - self.Kl, 0) - np.maximum(x - self.Ku, 0)
    
    def blackscholes_price(self, x, T, vol, rate=0.):
        price = black_scholes_call(x, self.Kl, T, vol, rate)
        price = price - black_scholes_call(x, self.Ku, T, vol, rate)
        return price
    
    def bachelier_price(self, x, T, vol, rate=None):
        if not (rate is None):
            raise NotImplementedError("Bachelier price with nonzero interest rates not implemented.")
        price = bachelier_call(x, self.Kl, T, vol)
        price = price - bachelier_call(x, self.Ku, T, vol)
        return price



class ButterflyOption(EuropeanOption):

    def __init__(self, strike: float, level: float) -> None:
        super().__init__()
        self.strike = float(strike)
        self.level = float(level)

    def payoff(self, x):
        return np.maximum(x - (self.strike - self.level), 0) - 2 * np.maximum(x - self.strike, 0) + np.maximum(x - (self.strike + self.level), 0)
    
    def blackscholes_price(self, x, T, vol, rate=0.):
        price = black_scholes_call(x, self.strike - self.level, T, vol, rate)
        price = price - 2 * black_scholes_call(x, self.strike, T, vol, rate)
        price = price + black_scholes_call(x, self.strike + self.level, T, vol, rate)
        return price

    def bachelier_price(self, x, T, vol, rate=None):
        if not (rate is None):
            raise NotImplementedError("Bachelier price with nonzero interest rates not implemented.")
        price = bachelier_call(x, self.strike - self.level, T, vol)
        price = price - 2 * bachelier_call(x, self.strike, T, vol)
        price = price + bachelier_call(x, self.strike + self.level, T, vol)
        return price



if __name__=='__main__':

    from processes import Bachelier, GBM
    
    S0 = 100
    sigma = 0.20
    T = 0.5
    times = np.linspace(0, T, 11)
    print(times, times.shape[0])

    ################ Butterfly Option #################
    butterfly_level = 20
    option = ButterflyOption(strike=S0, level=butterfly_level)

    print("######## test butterfly option with Bachelier model ##################")

    paths_nr = 1000000
    model = Bachelier(S0, 0, sigma * S0, times, paths_nr)
    model.simulate_paths(model.generate_noise())

    for i in range(1, times.shape[0]):
        print(f"------------ Butterfly test at time {times[i]:.4f} -------------")
        # mc_prices = []
        # an_prices = []
        mc_price = np.mean(option.payoff(model.paths[i]))
        # mc_prices.append(mc_call)
        mc_std = np.std(option.payoff(model.paths[i]))
        analytic_price = option.bachelier_price(S0, times[i], sigma * S0)
        # an_prices.append(analytic_call)
        mc_err = 2.33 * mc_std / np.sqrt(paths_nr)
        mc_test = np.absolute(mc_price - analytic_price) <= mc_err
        print(f"MC price {mc_price:.4f}. Analytic price: {analytic_price:.4f}. MC err: {mc_err:.4f}. MC test: {mc_test}")
        implied_vol = option.bach_implied_vol(analytic_price, S0, times[i])
        print(f"Implied bachelier volatility at time {times[i]:.4f}: {implied_vol:.4f}")

    
    print("\n\n######## test butterfly option with Black-Scholes model ##################")

    r = 0.05
    model = GBM(S0, r, sigma, times, paths_nr)
    model.simulate_paths(model.generate_noise())
    for i in range(1, times.shape[0]):
        print(f"------------ butterfly test at time {times[i]:.4f} -------------")
        mc_price = np.exp(-r * times[i]) * np.mean(option.payoff(model.paths[i]))
        mc_std = np.std(option.payoff(model.paths[i]))
        analytic_price = option.blackscholes_price(S0, times[i], sigma, r)
        mc_err = 2.33 * mc_std / np.sqrt(paths_nr)
        mc_test = np.absolute(mc_price - analytic_price) <= mc_err
        print(f"MC price {mc_price:.4f}. Analytic price: {analytic_price:.4f}. MC err: {mc_err:.4f}. MC test: {mc_test}")
        implied_vol = option.bs_implied_vol(analytic_price, S0, times[i], r)
        print(f"Implied bachelier volatility at time {times[i]:.4f}: {implied_vol:.4f}")



    ################### Bull Call Spread ##################
    strike_low = 90
    strike_high = 110
    option = BullCallSpread(strike_low, strike_high)

    print("\n\n\n######## test bull call spread option with Bachelier model ##################")

    paths_nr = 1000000
    model = Bachelier(S0, 0, sigma * S0, times, paths_nr)
    model.simulate_paths(model.generate_noise())

    for i in range(1, times.shape[0]):
        print(f"------------ Bull Call Spread test at time {times[i]:.4f} -------------")
        mc_price = np.mean(option.payoff(model.paths[i]))
        mc_std = np.std(option.payoff(model.paths[i]))
        analytic_price = option.bachelier_price(S0, times[i], sigma * S0)
        mc_err = 2.33 * mc_std / np.sqrt(paths_nr)
        mc_test = np.absolute(mc_price - analytic_price) <= mc_err
        print(f"MC price {mc_price:.4f}. Analytic price: {analytic_price:.4f}. MC err: {mc_err:.4f}. MC test: {mc_test}")
        implied_vol = option.bach_implied_vol(analytic_price, S0, times[i])
        print(f"Implied bachelier volatility at time {times[i]:.4f}: {implied_vol:.4f}")

    
    print("\n\n######## test bull call spread option with Black-Scholes model ##################")

    r = 0.05
    model = GBM(S0, r, sigma, times, paths_nr)
    model.simulate_paths(model.generate_noise())
    for i in range(1, times.shape[0]):
        print(f"------------ Bull Call Spread test at time {times[i]:.4f} -------------")
        mc_price = np.exp(-r * times[i]) * np.mean(option.payoff(model.paths[i]))
        mc_std = np.std(option.payoff(model.paths[i]))
        analytic_price = option.blackscholes_price(S0, times[i], sigma, r)
        mc_err = 2.33 * mc_std / np.sqrt(paths_nr)
        mc_test = np.absolute(mc_price - analytic_price) <= mc_err
        print(f"MC price {mc_price:.4f}. Analytic price: {analytic_price:.4f}. MC err: {mc_err:.4f}. MC test: {mc_test}")
        implied_vol = option.bs_implied_vol(analytic_price, S0, times[i], r)
        print(f"Implied bachelier volatility at time {times[i]:.4f}: {implied_vol:.4f}")