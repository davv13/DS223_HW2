"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *
import csv
import numpy as np
import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig
logger = logging.getLogger("Multi-Armed Bandit Application")

# Create a console handler with a higher log level and custom formatting
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials = 20000
epsilon = 0.2


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


class EpsilonGreedy(Bandit):
    def __init__(self, Rewards, exploration_rate=epsilon):
        self.exploration_rate = exploration_rate
        self.num_actions = len(Rewards)
        self.total_reward = 0
        self.action_rewards = []
        self.action_counts = np.zeros(self.num_actions)
        self.mean_rewards = np.zeros(self.num_actions)

    def __repr__(self):
        return "EpsilonGreedy(exploration_rate={}, num_actions={}, mean_rewards={})".format(self.exploration_rate, self.num_actions, self.mean_rewards)

    def pull(self):

        if np.random.rand() < self.exploration_rate:
            current_selected_action = np.random.choice(self.num_actions)
        else:
            current_selected_action = np.argmax(self.mean_rewards)

        reward = Bandit_Reward[current_selected_action]
        self.action_rewards.append((current_selected_action, reward, "EpsilonGreedy"))

        return current_selected_action, reward

    def update(self, current_selected_action, reward):
        self.action_counts[current_selected_action]+=1
        alpha = 1/self.action_counts[current_selected_action]
        self.mean_rewards[current_selected_action]+=alpha*(reward-self.mean_rewards[current_selected_action])
        self.total_reward+=reward

    def experiment(self, trials=NumberOfTrials):
        for cases in range(1, trials + 1):
            current_selected_action, reward = self.pull()
            self.update(current_selected_action, reward)
            self.exploration_rate = max(self.exploration_rate/cases, 0.01)

    def report(self):
        with open("all_rewards.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Bandit", "Reward", "Algorithm"])
            writer.writerows(self.action_rewards)

        best_reward = max(Bandit_Reward)
        mean_of_rewards = self.total_reward / sum(self.action_counts)
        mean_of_regrets = best_reward - mean_of_rewards
        print(f"Average Reward for EpsilonGreedy: {mean_of_rewards:.2f}")
        print(f"Average Regret for EpsilonGreedy: {mean_of_regrets:.2f}")

              

class ThompsonSampling(Bandit):
    def __init__(self, Rewards):
        self.num_actions = len(Rewards)
        self.alpha = np.ones(self.num_actions)
        self.beta = np.ones(self.num_actions)

        self.total_reward = 0
        self.action_rewards = []

    def __repr__(self):
        return f"ThompsonSampling (num_actions={self.num_actions}, alpha_parameter={self.alpha}, beta_parameter={self.beta})"

    def pull(self):
        trial_cases = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.num_actions)]
        current_selected_action = np.argmax(trial_cases)

        reward = Bandit_Reward[current_selected_action]
        self.action_rewards.append((current_selected_action, reward, "ThompsonSampling"))

        return current_selected_action, reward

    def update(self, current_selected_action, reward):
        if reward > 0:
            self.alpha[current_selected_action] += 1
        else:
            self.beta[current_selected_action] += 1
        self.total_reward += reward

    def experiment(self, trials=NumberOfTrials):
        for ii in range(trials):
            current_selected_action, reward = self.pull()
            self.update(current_selected_action, reward)

    def report(self):
        with open("all_rewards.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.action_rewards)

        best_reward = max(Bandit_Reward)
        mean_of_rewards = self.total_reward / (sum(self.alpha) + sum(self.beta) - 2 * self.num_actions)
        mean_of_regrets = best_reward - mean_of_rewards

        print(f"Average Reward for ThompsonSampling: {mean_of_rewards:.2f}")
        print(f"Average Regret for ThompsonSampling: {mean_of_regrets:.2f}")




class Visualization:
    def __init__(self, epsilon_rewards, ts_rewards):
        self.epsilon_rewards = epsilon_rewards
        self.ts_rewards = ts_rewards

        sns.set(style="whitegrid")  

    def plot_learning_process(self):
        epsilon_cum_rewards = self.calculate_cumulative_rewards(self.epsilon_rewards)
        thompson_cum_rewards = self.calculate_cumulative_rewards(self.ts_rewards)

        epsilon_mean_rewards = self.calculate_average_rewards(epsilon_cum_rewards)
        thompson_mean_rewards = self.calculate_average_rewards(thompson_cum_rewards)

        plt.figure(figsize=(16, 9))
        sns.lineplot(x=range(1, len(self.epsilon_rewards) + 1), y=epsilon_mean_rewards, label="Epsilon-Greedy")
        sns.lineplot(x=range(1, len(self.ts_rewards) + 1), y=thompson_mean_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Average Reward")
        plt.title("Learning Process of E-Greedy and Thompson Sampling")
        plt.legend()
        plt.show()

    def plot_cumulative_rewards(self):
        epsilon_cum_rewards = self.calculate_cumulative_rewards(self.epsilon_rewards)
        thompson_cum_rewards = self.calculate_cumulative_rewards(self.ts_rewards)

        plt.figure(figsize=(16, 9))
        sns.lineplot(x=range(1, len(self.epsilon_rewards) + 1), y=epsilon_cum_rewards, label="Epsilon-Greedy")
        sns.lineplot(x=range(1, len(self.ts_rewards) + 1), y=thompson_cum_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards of E-Greedy and Thompson Sampling")
        plt.legend()
        plt.show()

    def calculate_cumulative_rewards(self, rewards):
        cumulative_rewards = [reward for ii, reward, ii in rewards]
        cumulative_rewards_sum = [sum(cumulative_rewards[:i + 1]) for i in range(len(cumulative_rewards))]
        return cumulative_rewards_sum

    def calculate_average_rewards(self, cumulative_rewards):
        return [cumulative_reward / (i + 1) for i, cumulative_reward in enumerate(cumulative_rewards)]



def comparison(epsilon_rewards, ts_rewards):
    epsilon_cum_rewards = [reward for _, reward, _ in epsilon_rewards]
    thompson_cum_rewards = [reward for _, reward, _ in ts_rewards]

    # Plot cumulative rewards for Epsilon-Greedy and Thompson Sampling
    plt.figure(figsize=(16, 9))
    plt.plot(np.cumsum(epsilon_cum_rewards), label="Epsilon-Greedy")
    plt.plot(np.cumsum(thompson_cum_rewards), label="Thompson Sampling")
    plt.xlabel("Trials")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Rewards Comparison")
    plt.legend()
    plt.show()

    # Calculate and plot average rewards
    epsilon_avg_rewards = np.cumsum(epsilon_cum_rewards) / (np.arange(len(epsilon_cum_rewards)) + 1)
    thompson_avg_rewards = np.cumsum(thompson_cum_rewards) / (np.arange(len(thompson_cum_rewards)) + 1)

    plt.figure(figsize=(16, 9))
    plt.plot(epsilon_avg_rewards, label="Epsilon-Greedy")
    plt.plot(thompson_avg_rewards, label="Thompson Sampling")
    plt.xlabel("Trials")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards Comparison")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
    
    # Define Bandit_Reward, NumberOfTrials, and epsilon
    Bandit_Reward = [1, 2, 3, 4]
    NumberOfTrials = 20000
    epsilon = 0.2

    # Create EpsilonGreedy and ThompsonSampling instances
    epsilon_greedy = EpsilonGreedy(Bandit_Reward)
    thompson_sampling = ThompsonSampling(Bandit_Reward)

    # Run experiments
    epsilon_greedy.experiment()
    thompson_sampling.experiment()

    # Report results
    epsilon_greedy.report()
    thompson_sampling.report()

    # Create Visualization instance
    viz = Visualization(epsilon_greedy.action_rewards, thompson_sampling.action_rewards)

    # Plot learning process
    viz.plot_learning_process()

    # Plot cumulative rewards
    viz.plot_cumulative_rewards()

    comparison(epsilon_greedy.action_rewards, thompson_sampling.action_rewards)







