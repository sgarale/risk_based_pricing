import numpy as np


class GBM:
	def __init__(self, S0: float, mu: float, sigma: float, times: np.ndarray, paths_nr: int):
		self.S0 = float(S0)
		self.mu = float(mu)
		self.sigma = float(sigma)
		self.times = np.array(times)
		self.paths_nr = int(paths_nr)
		self.paths = None
		
	def generate_noise(self):
		x_rnd = np.random.normal(0,1,(self.times.shape[0] - 1, self.paths_nr))
		return x_rnd

	def simulate_paths(self, noises):

		S = self.S0
		paths = []
		paths.append(S * np.ones(self.paths_nr))

		for j in range(1, self.times.shape[0]):
			dt = self.times[j] - self.times[j - 1]
			drift = self.mu - 0.5 * np.power(self.sigma, 2)
			S *= np.exp(drift * dt + self.sigma * np.sqrt(dt) * (noises[j - 1][:]))
			paths.append(S.copy())

		self.paths = paths
		return
	


class Bachelier:
    """
    Class implementing a Bachelier process.

    Example:
        # setting parameters
        S0 = 100
        mu = 0.7
        sigma = 20
        paths_nr = 100000
        # setting times
        T = 0.5
        times = np.linspace(0, T, 11)
        print(times, times.shape[0])

        model = Bachelier(S0, mu, sigma, times, paths_nr)
        model.simulate_paths(model.generate_noise())
    """
    def __init__(self, S0: float, mu: float, sigma: float, times: np.ndarray, paths_nr: int):
        self.S0 = float(S0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.times = np.array(times)
        self.paths_nr = int(paths_nr)
        self.paths = None
		
    def generate_noise(self):
        x_rnd = np.random.normal(0,1,(self.times.shape[0] - 1, self.paths_nr))
        return x_rnd

    def simulate_paths(self, noises):

        S = self.S0
        paths = []
        paths.append(S * np.ones(self.paths_nr))

        for j in range(1, self.times.shape[0]):
            dt = self.times[j] - self.times[j - 1]			
            S += self.mu * dt + self.sigma * np.sqrt(dt) * noises[j - 1][:]
            paths.append(S.copy())

        self.paths = paths
        return



if __name__=='__main__':
	
    import options
    np.random.seed(18)

    print("######## tests Bachelier model ##################")
    S0 = 100
    mu = 0
    sigma = 20
    T = 0.5
    times = np.linspace(0, T, 11)
    print(times, times.shape[0])

    paths_nr = 1000000
    model = Bachelier(S0, mu, sigma, times, paths_nr)
    model.simulate_paths(model.generate_noise())
    for i in range(1, times.shape[0]):
        mc_mean = np.mean(model.paths[i])
        mc_std = np.std(model.paths[i])
        analytic_mean = S0 + mu * times[i]
        mc_err = 2.33 * mc_std / np.sqrt(paths_nr)
        mc_test = np.absolute(mc_mean - analytic_mean) <= mc_err
        print(f"Mean at time {times[i]:.4f}: {mc_mean:.4f}. Analytic mean: {analytic_mean:.4f}. MC err: {mc_err:.4f}. MC test: {mc_test}")


    strikes = np.arange(70, 135, 5)
    for i in range(1, times.shape[0]):
        print(f"------------ Call test at time {times[i]:.4f} -------------")
        for k in strikes:
            mc_call = np.mean(np.maximum(model.paths[i] - k, 0))
            mc_std = np.std(np.maximum(model.paths[i] - k, 0))
            analytic_call = options.bachelier_call(S0, k, times[i], sigma)
            mc_err = 2.33 * mc_std / np.sqrt(paths_nr)
            mc_test = np.absolute(mc_call - analytic_call) <= mc_err
            print(f"Strike {k}. MC price {mc_call:.4f}. Analytic price: {analytic_call:.4f}. MC err: {mc_err:.4f}. MC test: {mc_test}")


    print("\n\n######## tests GBM model ##################")
    S0 = 100
    r = -0.05
    sigma = 0.20
    T = 0.5
    times = np.linspace(0, T, 11)
    print(times, times.shape[0])

    paths_nr = 1000000
    model = GBM(S0, r, sigma, times, paths_nr)
    model.simulate_paths(model.generate_noise())
    for i in range(1, times.shape[0]):
        mc_mean = np.mean(model.paths[i])
        mc_std = np.std(model.paths[i])
        analytic_mean = S0 * np.exp(r * times[i])
        mc_err = 2.33 * mc_std / np.sqrt(paths_nr)
        mc_test = np.absolute(mc_mean - analytic_mean) <= mc_err
        print(f"Mean at time {times[i]:.4f}: {mc_mean:.4f}. Analytic mean: {analytic_mean:.4f}. MC err: {mc_err:.4f}. MC test: {mc_test}")


    strikes = np.arange(70, 135, 5)
    for i in range(1, times.shape[0]):
        print(f"----------- Call test at time {times[i]:.4f} ---------------")
        for k in strikes:
            mc_call = np.exp(- r * times[i]) * np.mean(np.maximum(model.paths[i] - k, 0))
            # mc_prices.append(mc_call)
            mc_std = np.std(np.maximum(model.paths[i] - k, 0))
            analytic_call = options.black_scholes_call(S0, k, times[i], sigma, r)
            # an_prices.append(analytic_call)
            mc_err = 2.33 * mc_std / np.sqrt(paths_nr)
            mc_test = np.absolute(mc_call - analytic_call) <= mc_err
            print(f"Strike {k}. MC price {mc_call:.4f}. Analytic price: {analytic_call:.4f}. MC err: {mc_err:.4f}. MC test: {mc_test}")