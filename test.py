import numpy as np
import yaml

from unityagents import UnityEnvironment

from utils.utils import load_agent


np.set_printoptions(precision=3, suppress=True)


def test(env, agent_path, game_play):

    multi_agent = load_agent(path=agent_path)
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]

    observations = env_info.vector_observations
    scores = np.zeros(multi_agent.agent_count)

    for i in range(game_play):
        while True:
            actions = multi_agent.act(observations)
            env_info = env.step(actions)[brain_name]
            next_observations = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            observations = next_observations

            print('\rCurrent score %s' % scores, end='')

            if np.any(dones):
                continue

    print('\rDone! with score %s' % scores)


if __name__ == "__main__":

    yaml_path = 'examples/tennis/config.yaml'

    with open(yaml_path, 'r') as f:
        cfg = yaml.load(f)

    env_filename = cfg['test_env_filepath']
    model_path = cfg['model_filepath']

    env = UnityEnvironment(file_name=env_filename)
    test(env, model_path, game_play=4)

    env.close()
