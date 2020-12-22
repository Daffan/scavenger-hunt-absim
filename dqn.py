import os
import gym
import json
import torch
import pprint
import argparse
from datetime import datetime
import numpy as np
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, ShmemVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer

import env

CONFIG_PATH = "config/default.json"

def load_args(config_path=CONFIG_PATH):
    # Load the configurations of env and training from a config file
    with open(config_path, "r") as f:
        args = json.load(f)
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    training_args = AttrDict(**args["training_args"])
    env_args = AttrDict(**args["env_args"])
    return training_args, env_args

def test_dqn(args=load_args()):
    # load config
    env_args = args[1]
    args = args[0]
    # load environments
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task, **env_args) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task, test=True, **env_args) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.layer_num, args.state_shape,
              args.action_shape, args.device,  # dueling=(1, 1)
              ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # learning schedule
    if args.lr_schedule=="linear":
        lr_lambda = lambda epoch: (1-float(epoch)/args.epoch)
    else:
        lr_lambda = lambda epoch: 1 # constant lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    policy = DQNPolicy(
        net, optim, args.gamma, args.n_step,
        target_update_freq=args.target_update_freq)
    # buffer
    if args.prioritized_replay > 0:
        buf = PrioritizedReplayBuffer(
            args.buffer_size, alpha=args.alpha, beta=args.beta)
    else:
        buf = ReplayBuffer(args.buffer_size)
    # collector
    train_collector = Collector(policy, train_envs, buf)
    test_collector = Collector(policy, test_envs)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size)
    # log
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    log_path = os.path.join("log", args.task, 'dqn', dt_string)
    writer = SummaryWriter(log_path)
    copyfile(CONFIG_PATH, os.path.join(log_path, "default.json"))

    def save_fn(policy):
        torch.save(policy.model.state_dict(), os.path.join(log_path, 'policy.pth'))

    def train_fn(epoch, env_step):
        policy.set_eps(max(args.final_eps, args.init_eps*(1-2*(epoch-1)/(args.epoch-1))))

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)
        scheduler.step()

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        save_fn=save_fn, writer=writer, log_interval=10)

if __name__ == '__main__':
    test_dqn(load_args())
