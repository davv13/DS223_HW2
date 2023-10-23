# DS223_HW2

A/B Testing
Scenario
You have four advertisement options (bandits), and your task is to design an experiment using 
Epsilon Greedy and Thompson Sampling.
Design of Experiment
A bandit class has already been created for you. It is an abstract class with abstract methods. You 
must not exclude anything from the Bandit() class. However, you can add more stuff if you 
need.

Bandit_Reward=[1,2,3,4]
NumberOfTrials: 20000

1. Create a Bandit Class

2. Create EpsilonGreedy() and ThompsonSampling() classes and methods (inherited
from Bandit()).
2.1. Epsilon-greedy: 
2.1.1. decay epsilon by 1/t
2.1.2. design the experiment
2.2. Thompson Sampling 
2.2.1. design with known precision
2.2.2. design the experiment 

3. Report:
1. Visualize the learning process for each algorithm (plot1())
2. Visualize cumulative rewards from E-Greedy and Thompson Sampling.
3. Store the rewards in a CSV file ({Bandit, Reward, Algorithm})
4. Print cumulative reward
5. Print cumulative regret

Note the values of epsilon and precision are up to you to decide.
