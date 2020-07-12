import numpy as np
import sobol_seq as sb
from scipy import stats
import math
import matplotlib.pyplot as plt


# This function calculates future payoff of the asian option based on arithmetic average of the price path
def payoff_calc(price_array, X):
    payoff = np.maximum(0, np.mean(price_array) - X)
    return payoff


def pv_calc(FV, r, T):
    return FV * np.exp(-r * T)


# This function calculates 95% credit interval for the expected value of a random variable, given a sample
def CI_calc(s_array):
    X_bar = np.mean(s_array)
    Upper_b = X_bar + (np.sqrt(np.var(s_array)) * stats.norm.ppf(0.975)) / np.sqrt(len(s_array))
    Lower_b = X_bar - (np.sqrt(np.var(s_array)) * stats.norm.ppf(0.975)) / np.sqrt(len(s_array))
    return np.array([Lower_b, Upper_b])


r = 0.1
sigma = 0.2
T = 2  # This is the length of time period (in years)
n_steps = 20  # number of time steps in each path
dt = T / n_steps
S_0 = 50
x_price = 40    # this is the exercise price or K
Max_n_of_p = 5000   # maximum number of paths (no_of_paths means number of samples in each simulation)

tol = 0.5

mean_pv_payoffs = np.zeros(int(np.ceil(Max_n_of_p / 10)))
credit_intervals = np.array([None, None])

for no_of_paths in range(10,Max_n_of_p+1,10):

    present_payoffs = np.zeros(no_of_paths)

    for k in range(no_of_paths):
        price_steps = np.zeros(n_steps)
        price_steps[0] = S_0
        sob_seq = np.array([])
        for n in range(math.ceil(n_steps / 40)):
            sob_seq = np.concatenate((sob_seq, sb.i4_sobol(40, np.random.randint(1, 100))[0]))
        for i in range(1, n_steps):
            epsilon_s = stats.norm.ppf(sob_seq)
            price_steps[i] = price_steps[i-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * epsilon_s[i] * np.sqrt(dt))
        present_payoffs[k] = pv_calc(payoff_calc(price_steps, x_price), r, T)
    mean_pv_payoffs[int(no_of_paths/10 - 1)] = np.mean(present_payoffs)
    credit_intervals = np.row_stack((credit_intervals, CI_calc(present_payoffs)))
    print(mean_pv_payoffs[int(no_of_paths / 10 - 1)])
    print(credit_intervals[int(no_of_paths / 10 - 1)])


x_axis1 = range(10, Max_n_of_p + 1, 10)
plt.plot(x_axis1, mean_pv_payoffs)
plt.plot(x_axis1, credit_intervals[1:, 0], 'g--', lw=0.5, label='Upper and lower bound of Confidence interval')
plt.plot(x_axis1, credit_intervals[1:, 1], 'g--', lw=0.5)
plt.xlabel("No. of Samples in simulation")
plt.ylabel("Estimated option price")
plt.title("QMC (Sobol Sequence) Method")
plt.legend()
plt.show()

plt.hist(present_payoffs, 200)
plt.title("Histogram of present value of payoffs in QMC (Sobol) Method")
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.show()

CI_length = credit_intervals[1:, 1] - credit_intervals[1:, 0]
print(CI_length[-1])
for i in range(0,len(CI_length)):
    if CI_length[i] <= tol:
        print(i)
        break

plt.plot(x_axis1, CI_length)
plt.axhline(tol, ls='--', c='r')
plt.xlabel("No. of Samples in simulation")
plt.ylabel("Length of confidence interval")
plt.title("Accuracy of Estimation in QMC (Sobol)")
plt.show()


#plt.hist(present_payoffs, 200)
#plt.show()
#final_prices = [i[-1] for i in price_paths]
#plt.hist(final_prices, 200)
#plt.show()
