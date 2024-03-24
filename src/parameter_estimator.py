import numpy as np

true_parameter = np.array([0.9, -0.8, -0.2])

def get_reward(w): 
    return -np.sum((w - true_parameter)**2)

population_size = 50
std_noise = 0.1
alpha = 0.001

npop = 50      # population size
sigma = 0.1    # noise standard deviation
alpha = 0.001  # learning rate

w_est = np.random.randn(3)

for i in range(200):
    candidates = np.random.randn(population_size, 3)
    rewards = np.zeros(population_size)
    for j in range(population_size):
        w_try = w_est + std_noise*candidates[j]
        rewards[j] = get_reward(w_try)
    A = (rewards - np.mean(rewards)) / np.std(rewards)
    w_est = w_est + alpha/(npop*sigma) * np.dot(candidates.T, A)
    print(w_est)
