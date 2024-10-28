## Discrete approximation of risk-based prices under volatility uncertainty
____________________________

This repository contains the implementation of the numerical examples in ["Discrete approximation of risk-based prices under volatility uncertainty"](link)

### Requirements
The file [requirements.yml](requirements.yml) contains minimal requirements for the environment necessary to run the scripts of this repository. The channel might change depending on the architecture of the machine.

### Main classes and Notebooks

The code is written using an object oriented philosophy, so that generalizations should be feasible. All the main classes, the scripts and the notebooks are commented for a minimal guidance.

Here a list of the main scripts and notebooks:
- [expectations.py](expectations.py) contains several classes implementing linear and nonlinear expectations. Also contains classes implementing the entropic risk measure.
- [iterators.py](iterators.py) contains the functions performing the iterations of different one-step pricing operators (certainty case, uncertainty case, G-expectation).
- [options.py](options.py) contains classes implementing some European options. Each class provides the payoff function and closed formulas for the Black-Scholes and Bachelier prices.

- [bf_bin_conv.ipynb](no_uncertainty/bf_bin_conv.ipynb) Produces the plot of Figure 1 showing the convergence of the binomial model to the Bachelier prices. It also saves the dictionary of the iterations.
- [bf_diff_models.ipynb](no_uncertainty/bf_diff_models.ipynb) Produces the plots of Figure 2 and saves the dictionaries of the iterations.
- [bf_sub_unc.ipynb](uncertainty/bf_sub_unc.ipynb) Produces the plots of Figure 3 and saves the dictionaries of the iterations.
- [bf_sub_alpha.ipynb](uncertainty/bf_sub_alpha.ipynb) Computes the iterations for the uncertain binomial model and the butterfly option. Its output is used to produce Figure 4.
- [butterfly_sub_alpha.ipynb](elaborations/butterfly_sub_alpha.ipynb) Produces the plot of Figure 4. It requires the dictionaries of the iterations.
- [bf_sub_alpha_bid](uncertainty/bf_sub_alpha_bid.ipynb) Computes the iterations for the bid price of the butterfly option and produces Figure 5(a).
- [bf_G_exp.ipynb](uncertainty/bf_G_exp.ipynb) Computes the iterations for the G-expectation associated to a butterfly option. Its output is used to produce Figure 4 and Figure 5.
- [butterfly_bid_ask.ipynb](elaborations/butterfly_bid_ask.ipynb) Produces the plot of Figure 5(b). It requires the dictionaries of the iterations.

The results of the iterations for the pricing functionals are always saved as pickle files to allow for later elaborations. The folder [output](output) contains the pickle files necessary to produce the plots in the paper.

Execution time may vary depending on the machine.

#### Version 0.9

### License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details