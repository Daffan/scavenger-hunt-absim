import numpy as np
from itertools import permutations, product
from dqn import load_args
from tianshou.utils.net.common import Net
import torch
import gym
from os.path import join, dirname

class dqn():
    def __init__(self, env, policy):
        self.N = env.map.get_node_num()
        self.env = env

        config_path = join(dirname(policy), "default.json")
        args, _ = load_args(config_path)
        # model
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        self.net = Net(args.layer_num, state_shape,
                       action_shape, args.device,  # dueling=(1, 1)
                       ).to(args.device)
        state_dict = torch.load(policy)
        self.net.load_state_dict(state_dict)

    def next_node(self, obs):
        out, _ = self.net(np.array([obs]))
        out = out.detach().cpu()[0]
        return np.argmax(out)

class probability():
    def __init__(self, env):
        self.N = env.map.get_node_num()
        self.env = env

    def next_node(self, obs):
        return np.argmax(obs[:self.N])

class proximity():
    def __init__(self, env):
        self.N = env.map.get_node_num()
        self.env = env

    def next_node(self, obs):
        cl = self.env.map.get_current_loc()
        dist = [obs[self.N+i] if obs[i]>0 else np.inf for i in range(self.N)]
        return np.argmin(dist)

class probability_proximity():
    def __init__(self, env):
        self.N = env.map.get_node_num()
        self.env = env

    def next_node(self, obs):
        curr_loc = self.env.map.get_current_loc()
        weight = [obs[i]/obs[i+self.N] if i!= curr_loc else 0\
                 for i in range(self.N)]
        return np.argmax(weight)

class optimal():
    def __init__(self, env):
        self.N = env.map.get_node_num()
        self.env = env

    def next_node(self, node):
        node_to_visit = [n for i, n in enumerate(self.env.map.obj_loc)\
                         if self.env.map.obj_list[i]]
        paths = list(permutations(node_to_visit))
        path_cost = [self.compute_path_cost(p) for p in paths]
        idx = np.argmin(path_cost)
        path = paths[idx]
        return path[0]

    def compute_path_cost(self, path):
        path = [self.env.map.get_current_loc()] + list(path)
        c = 0
        for i in range(len(path)-1):
            c += self.env.map.get_cost(path[i], path[i+1])
        return c

class bayesian():
    def __init__(self, env):
        self.N = env.map.get_node_num()
        self.env = env

    def compute_path_cost(self, path):
        path = [self.env.map.get_current_loc()] + list(path)
        c = 0
        for i in range(len(path)-1):
            c += self.env.map.get_cost(path[i], path[i+1])
        return c

    def expected_path_cost(self, path):
        # get product of possible node position of 
        node_pemut = []
        for od in self.env.map.cur_distrs:
            node = []
            for i, p in enumerate(od):
                if p>0:
                    node.append((i, p))
            if node:
                node_pemut.append(node)
        node_pemut = list(product(*node_pemut))
        expected_cost = 0

        for node in node_pemut:
            prob = 1
            node_list = []
            for i, p in node:
                prob *= p
                node_list.append(i)
            for i, n in enumerate(path):
                if n in node_list:
                    # remove all the occurrance of n
                    node_list = list(filter((n).__ne__, node_list))
                if len(node_list)==0:
                    break
            cost = self.compute_path_cost(path[:i+1])
            expected_cost += cost * prob
            # print(node, cost)
        # print(path, i, expected_cost)
        return expected_cost

    def next_node(self, obs):
        prob = obs[:self.N]
        node_to_visit = [i for i, p in enumerate(prob) if p>0]
        path_list = list(permutations(node_to_visit))
        min_expected_cost = np.inf
        min_path = None

        count = 1
        for path in path_list:
            count += 1
            expected_cost = self.expected_path_cost(list(path))
            if expected_cost < min_expected_cost:
                min_path = path
                min_expected_cost = expected_cost
        # print(list(min_path), prob)
        # print(self.env.map.obj_loc, self.env.map.obj_list, "\n")
        return list(min_path)[0]


