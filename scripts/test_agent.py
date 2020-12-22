import sys
from os.path import join, abspath, dirname
sys.path.append(dirname(dirname(join(abspath(__file__)))))
import env
import gym
import numpy as np
from agent import bayesian, dqn, probability, proximity, probability_proximity, optimal
from dqn import load_args
import warnings
warnings.simplefilter('error')

CONFIG_PATH = "config/default.json"

AGENT_DICT = {
    "prob": probability,
    "prox": proximity,
    "prob_prox": probability_proximity,
    "optimal": optimal,
    "dqn": dqn,
    "bayes": bayesian
}

def test_agent(fname, agent, avg=100, seed=43):
    _, env_args = load_args(CONFIG_PATH)
    if fname is not None:
        # if map is specified, use the map without random map
        env_args["fname"] = fname
        env_args["random_map"] = False
    env = gym.make("ScavengerHuntMap-v0", **env_args)
    env.seed(seed)
    dist_list = []
    a = agent(env)
    for i in range(avg):
        print("Running %d/%d" %((i+1), avg), end="\r")
        obs = env.reset()
        done = False
        dist = 0
        while not done:
            act = a.next_node(obs)
            cl = env.env.map.get_current_loc()
            obs, _, done, info = env.step(act)
            dist += info["cost"]
        dist_list.append(dist)
    return sum(dist_list)/avg, np.std(dist_list)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", help="agent to test.", type=str, default="prob_prox"
    )
    parser.add_argument(
        "--map", help="path to the map. if not specified, config from /config/default.json will be used", type=str, default=None
    )
    parser.add_argument(
        "--avg", help="number of averaging.", type=int, default=100
    )
    parser.add_argument(
        "--policy", help="path of the policy to test. if specified, env config will be the same as training", type=str, default=None
    )
    parser.add_argument(
        "--seed", help="random seed.", type=int, default=43
    )

    args = parser.parse_args()
    print("Test %s agent %d times on map: %s" %(args.agent, args.avg, args.map))
    agent = AGENT_DICT[args.agent] if args.agent!= "dqn"\
            else lambda env: AGENT_DICT[args.agent](env, args.policy)
    if args.policy:
        # load the env config from training
        CONFIG_PATH = join(dirname(args.policy), "default.json")
    avg, std = test_agent(args.map, agent, args.avg, args.seed)
    print("Average distance: %.2f" %(avg))
    print("Standard deviation: %.2f" %(std))
    

