# Reacher

![reacher](resources/play.gif)

## Project details

This project contains a solution to the second project of Udacity Deep Reinforcement Learning.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

It is helpful to check the repository below for details.
* https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control

## Getting started
### Prerequired
* Python 3.6
* Unity

And then to install python dependencies. 

    pip install -r requirements.txt

Then you should be able to run `jupyter notebook` and view `Continuous_Control.ipynb`. 

The code for the Model and Agent are in `model.py` and `agent.py`, respectively.

## Instructions

Run each cell of `Continuous_Control.ipynb`.

You can also run `run.py` to pop up the Unity Agent directly and check the behavior with the already trained weight.

    python run.py