# Multi-Agent DDPG on ml-agents environment
- This repository implements maddpg (Multi-Agent Deep Deterministic Policy Gradients)[1] for solving multi-agent mixed cooperative-competitive environments. 

# Environment
![tennis screen](images/tennis-screenshot.png?raw=true)

The goal of this environment is control two agents to bounce a ball over a net with rackets and keep the ball in play. 

- Reward
  - each agent receives its own score. 
  - +0.1 if an agent hits a ball over the net. 
  - -0.01 if an agent lets a ball hit the ground or hits the ball out of bounds. 

- Observation space
  - each agent receives its own local observation. 
  - observation space of size 24 (of 3 frames)
    - single frame observation space size is 8.  
      - size 4 for a ball position and velocity.
      - size 4 for a racket position and velocity.
- Action space
  - each agent is allowed to navigate and jump around in their court region. 
  - continuous action space of size 2 
    - moving toward and bacward 
    - jumping up and down 
  
- Solving condition 
  - the task is episodic, the agents must get an average score of +0.5 over 100 consecutive episodes. 
  - score calculation 
     1. add up each agent score separately during a game play as described in the reward section
        - score1 and score2 for respective agents 
     2. score for each episode = max(score1, score2)
     3. average over 100 consecutive episodes

## Requirements
- python3
- ml-agents 0.4.0b
  - follow this [guide](https://github.com/Unity-Technologies/ml-agents/blob/0.4.0b/docs/Installation.md)
  - make sure to switch to 0.4.0b branch as follows before performing the installation 
    - `git checkout -b 0.4.0b 0.4.0b`
  - if 0.4.0 is installed, one can upgrade to 0.4.0b as follows
    - `pip install -U .`
- Tennis and Soccer environments  
  - follow this [guide](https://github.com/udacity/deep-reinforcement-learning/blob/master/p3_collab-compet/README.md)
- pytorch >= 0.4.1

## How to use? 
- Training a multi-agent
  - Edit the `env_filename`, `train_config` and `agent_config` in the `train.py` for different configurations.  
  - Run `python3 train.py` to train the multi-agent. 
- Evaluate the multi-agent 
  - Edit the `env_filename` and `agent_path` in the `test.py` for different configurations. 
  - Pretrained models are located inside the `examples/<environment>/model` directory.
  - Run `python3 test.py` to evaluate the trained models. 

## References
[1] [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)
