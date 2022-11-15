import time
import copy
from multiprocessing import Pipe, Process

import gym
import pickle
import cloudpickle
import numpy as np

############################################
############################################


def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states


def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

    def __call__(self):
        return self.x()


def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    step = 0
    while True:
        cmd, data = remote.recv()

        if cmd == 'step':
            action, max_path_length = data
            ob, reward, done, info = env.step(action)
            step += 1
            if done or step == max_path_length:
                step = 0
                ob = env.reset()
            remote.send((ob, reward, done, info))

        elif cmd == 'reset':
            step = 0
            remote.send(env.reset())

        elif cmd == 'render':
            remote.send(env.render())

        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError


class SubprocVecEnv:
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.no_of_envs = len(env_fns)
        self.remotes, self.work_remotes = \
            zip(*[Pipe() for _ in range(self.no_of_envs)])
        self.ps = []

        for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
            proc = Process(target=worker,
                           args=(wrk, rem, CloudpickleWrapper(fn)))
            self.ps.append(proc)

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions, max_path_length=None):
        if self.waiting:
            raise Exception
        self.waiting = True

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', (action, max_path_length)))

    def step_wait(self):
        if not self.waiting:
            raise Exception
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions, max_path_length=None):
        self.step_async(actions, max_path_length)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def make_mp_envs(env_id, num_env, seed, start_idx = 0):
    def make_env(rank):
        def fn():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
        return fn
    return SubprocVecEnv([make_env(i + start_idx) for i in range(num_env)])


def sample_trajectories_multiproc(
    envs: SubprocVecEnv,
    policy,
    max_path_length: int,
    min_timesteps_per_batch: int
):
    # no render
    obs = envs.reset()
    # one path list per env
    paths = [
        {
            'obs': [],
            'acs': [],
            'rewards': [],
            'next_obs': [],
            'terminals': []
        }
        for _ in range(envs.no_of_envs)
    ]

    complete_paths = []
    timesteps_this_batch = 0

    while timesteps_this_batch < min_timesteps_per_batch:

        acs = policy.get_action(obs)
        for ob, ac, path in zip(obs, acs, paths):
            path['obs'].append(ob)
            path['acs'].append(ac)

        obs, rews, dones, _ = envs.step(acs, max_path_length)
        for i, (next_ob, rew, done, path) in enumerate(zip(obs, rews, dones, paths)):
            path['next_obs'].append(next_ob)
            path['rewards'].append(rew)
            path['terminals'].append(done)
            if done:
                complete_paths.append(Path(**path, image_obs=[]))
                timesteps_this_batch += len(path['obs'])
                paths[i] = {
                    'obs': [],
                    'acs': [],
                    'rewards': [],
                    'next_obs': [],
                    'terminals': []
                }

    return complete_paths, timesteps_this_batch


def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # initialize env for the beginning of a new rollout
    ob = env.reset()  # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)  # HINT: query the policy's get_action function
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = done or len(obs) >= max_path_length
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # TODO: make sample_trajectory use parallel environments, sample rollouts until we have enough complete episodes (enough means min_timesteps_per_batch)
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        timesteps_this_batch += get_pathlength(paths[-1])

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1
    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data


if __name__=="__main__":
    envs = make_mp_envs('Ant-v2', 3, 3)
    print(envs.reset())
    # gym.make('Ant-v2')