import numpy as np
from unityagents import UnityEnvironment

from utils.utils import load_agent


np.set_printoptions(precision=3, suppress=True)


def test(env, agent_path):

    multi_agent = load_agent(path=agent_path)
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]

    observations = env_info.vector_observations
    scores = np.zeros(multi_agent.agent_count)

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
            break

    print('\rDone! with score %s' % scores)


if __name__ == "__main__":

    env_filename = 'examples/tennis/Tennis.app'

    agent_path = 'examples/tennis/model/checkpoint_104.pth'

    env = UnityEnvironment(file_name=env_filename)
    test(env, agent_path)

    env.close()
