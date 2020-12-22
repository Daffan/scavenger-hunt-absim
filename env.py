import numpy as np
import gym
from gym.envs.registration import register
from gym.spaces import Box, Discrete
from map import load_map, generate, reset_distribution
from os import path
import random

class ScavengerHunt(gym.Env):
    """
    Scavenger Hunt environment. Reward is (# objects being found) X object_reward - cost / maximum cost range / sqrt(2)
    The second term in the reward is normalized to [-1, 0]. Observation is the probability of finding at least one object
    params:
        fname [str]: path to the ./dat map file
        random_map [bool]: True if the map is resampled at an fixed interval of number of episode
        keep_map [bool]: valid only when random_map is True. Keep the graph of the map the same, only changes the distribution of the objs
        test [bool]: when test is True, the reward is traveled distance, a better metrics to see when doing the test
        switch_interval [int]: switch the map after switch_interval number of episode
        object_reward [float]: a positive reward when an object is found
        node_ranges, cost_range, objects_range, occurrences_range: settings for generating a random map
    """
    def __init__(self, fname="maps/default.dat", random_map=True,\
                 keep_map=False, test=False,\
                 switch_interval=10, object_reward=1,\
                 node_ranges=[8,8], cost_range=[10,500],
                 objects_range=[4,4], occurrences_range=[1,4]):

        self.fname = fname
        self.random_map = random_map
        self.switch_interval = switch_interval
        self.object_reward = object_reward
        self.test = test
        self.keep_map = keep_map
        self.node_ranges = node_ranges
        self.cost_range = cost_range
        self.objects_range = objects_range
        self.occurrences_range = occurrences_range
        self.max_step = 10 * node_ranges[-1]

        if not path.exists(fname):
            self.reset_map()
        self.map = load_map(fname)
        self.map.reset()

        self.N = self.map.get_node_num()
        self.ep_count = self.switch_interval+1

        self.action_space = Discrete(self.N)
        self.observation_space = Box(low=0, high=1, shape=(self.N,), dtype=float)

    def reset(self):
        if self.ep_count>self.switch_interval and self.random_map:
            self.reset_map()
            self.map.reset()
            self.ep_count = 0
        self.ep_count += 1
        self.map.reset()
        self.step_count = 0
        return self.map.get_find_at_least_one_prob()

    def step(self, action):
        curr_loc = self.map.get_current_loc()
        self.step_count += 1
        cost, found = self.map.move(action)
        obs = self.map.get_find_at_least_one_prob()
        rew = found*self.object_reward - cost/(self.cost_range[-1]*np.sqrt(2))
        done = sum(self.map.get_object_list())==0
        if self.step_count>=self.max_step:
            done = True
        if action==curr_loc:
            rew = -1
        if self.test:
            rew = -cost
        info = {"found": found, "cost": cost}
        return obs, rew, done, info

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset_map(self):
        if not self.keep_map:
            generate(self.fname, self.node_ranges, self.cost_range,
                    self.objects_range, self.occurrences_range)
            self.map = load_map(self.fname)
        else:
            self.map = reset_distribution(self.fname, self.node_ranges,
                                          self.cost_range, self.objects_range,
                                          self.occurrences_range)
    def close(self):
        pass

class WithMap(gym.Wrapper):
    """
    A wrapping the ScavengerHunt env. Cost map is attached to the original observation.
    Cost to the current node is set to be 1 (the largest cost). 
    """
    def __init__(self, **kwargs):
        super(WithMap, self).__init__(ScavengerHunt(**kwargs))
        self.observation_space = Box(low=0, high=1, shape=(2*self.N,), dtype=float)
        self.max_cost = self.cost_range[-1]/np.sqrt(2)

    def reset(self):
        obs = self.env.reset()
        obs = list(obs) + [c/self.max_cost for c in self.env.map.get_cost_map()]
        curr_loc = self.env.map.get_current_loc()
        obs[self.N+curr_loc] = 1
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = list(obs) + [c/self.max_cost for c in self.env.map.get_cost_map()]
        curr_loc = self.env.map.get_current_loc()
        obs[self.N+curr_loc] = 1
        return obs, rew, done, info

register(id="ScavengerHunt-v0", entry_point="env:ScavengerHunt")
register(id="ScavengerHuntMap-v0", entry_point="env:WithMap")

if __name__ == "__main__":
    import random

    env = WithMap(fname="maps/test.dat")
    env.reset() 

    N = env.N
    done = False

    while not done:
        action = random.choice(range(N))
        obs, rew, done, info = env.step(action)

        print("node: %d" %(action))
        print(obs, rew, done, info, "\n")
    env.close()

