import numpy as np
from collections import deque
from unityagents import UnityEnvironment

from maddpg.multi_agent import MultiAgent
from utils.utils import save_agent


def train(environment,
          train_config,
          agent_config,
          print_every,
          solving_score,
          random_seed=None):

    n_episodes = train_config['n_episodes']
    max_t = train_config['max_t']
    ou_noise = train_config['ou_noise_start']
    ou_noise_decay_rate = train_config['ou_noise_decay_rate']

    # get the default brain
    env = environment
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # Initialize our agent
    observation_size = brain.vector_observation_space_size * \
        brain.num_stacked_vector_observations
    action_size = brain.vector_action_space_size
    agent_count = len(env_info.agents)

    multi_agent = MultiAgent(
        agent_count=agent_count,
        observation_size=observation_size,
        action_size=action_size,
        train_config=train_config,
        agent_config=agent_config,
        seed=random_seed
    )

    all_train_scores = []
    all_val_scores = []
    solve_epi = 0

    val_scores_window = deque(maxlen=print_every)
    train_scores_window = deque(maxlen=print_every)

    for i_episode in range(1, n_episodes + 1):

        train_scores = train_episode(multi_agent, brain_name, max_t, ou_noise)
        val_scores = validate_episode(multi_agent, brain_name, max_t)

        ou_noise *= ou_noise_decay_rate

        train_scores = np.max(train_scores)
        train_scores_window.append(train_scores)
        all_train_scores.append(train_scores)

        val_scores = np.max(val_scores)
        val_scores_window.append(val_scores)
        all_val_scores.append(val_scores)

        print('\rEpisode {}\tAverage Training Score: {:.3f}'
              '\tValidation Score: {:.3f}'
              .format(
                  i_episode,
                  np.mean(train_scores_window),
                  np.mean(val_scores_window)
              ), end='')

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Training Score: {:.3f}'
                  '\tValidation Score: {:.3f}'
                  .format(
                      i_episode,
                      np.mean(train_scores_window),
                      np.mean(val_scores_window)))

        if np.mean(train_scores_window) >= solving_score and solve_epi == 0:
            print('\nEnvironment solved in {:d} episodes!'
                  '\tAverage Training Score: {:.3f}'
                  '\tValidation Score: {:.3f}'
                  .format(
                      i_episode,
                      np.mean(train_scores_window),
                      np.mean(val_scores_window)))

            solve_epi = i_episode

    return multi_agent, all_train_scores, all_val_scores, solve_epi


def train_episode(multi_agent, brain_name, max_t, ou_noise):

    env_info = env.reset(train_mode=True)[brain_name]
    obs = env_info.vector_observations
    multi_agent.reset()
    scores = np.zeros(multi_agent.agent_count)

    for _ in range(max_t):
        actions = multi_agent.act(obs, noise=ou_noise)
        brain_info = env.step(actions)[brain_name]
        next_obs = brain_info.vector_observations
        rewards = np.asarray(brain_info.rewards)
        dones = np.asarray(brain_info.local_done)

        multi_agent.step(obs, actions, rewards, next_obs, dones)
        obs = next_obs
        scores += rewards

        if np.any(dones):
            break

    return scores


def validate_episode(multi_agent, brain_name, max_t):
    """Validation execute an episode without noise
    """

    fast_simulation = True
    env_info = env.reset(train_mode=fast_simulation)[brain_name]
    obs = env_info.vector_observations
    scores = np.zeros(multi_agent.agent_count)

    for _ in range(max_t):
        actions = multi_agent.act(obs)
        env_info = env.step(actions)[brain_name]
        next_obs = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        obs = next_obs
        scores += rewards

        if np.any(dones):
            break

    return scores


if __name__ == '__main__':

    base_path = 'examples/tennis'
    env_filename = base_path + '/Tennis_Linux_NoVis/Tennis.x86_64'

    session = 104
    checkpoint_save_path = base_path + ('/model/checkpoint_%s.pth' % session)

    env = UnityEnvironment(file_name=env_filename)

    train_config = {
        # training loop
        'n_episodes': 2500,
        'max_t': 10000,
        'mini_batch_size': 256,
        'update_every': 2,

        # optimizers
        'actor_optim_params': {
            'lr': 1e-4
        },
        'critic_optim_params': {
            'lr': 1e-3,
        },

        # noise
        'ou_noise_start': 1.0,
        'ou_noise_decay_rate': 1.0,

        # maddpg
        'soft_update_tau': 5e-3,
        'discount_gamma': 0.95,

        # replay memory
        'buffer_size': 1000000
    }

    agent_config = {
    }

    multi_agent, _, _, _ = train(
        environment=env,
        train_config=train_config,
        agent_config=agent_config,
        print_every=100,
        solving_score=0.5,
        random_seed=0
    )

    env.close()

    save_agent(multi_agent, checkpoint_save_path)
