# scavenger-hunt-absim
A light implementation of Scavenger Hunt under graph representation. Registered gym environment `ScavengerHunt-v0` takes probability of finding at least one object as the observation, and takes actions of visiting one of the nodes. `ScavengerHuntMap-v0` has additional cost map, which is represented as edge cost to all the nodes respect to the current node. 

## Train a DQN model
* The program uses the implementation of DQN from [tianshou](https://github.com/thu-ml/tianshou). First install tianshou:
`pip install tianshou==0.3.0`
* Check the configuration `config/default.json`. The config file specifies all the training and environment parameters. 
* Train a DQN policy:
`python3 dqn.py`
* Check the log file under `log/`. You will find tensorboard log, saved policy and a copy of training configuration. 

## Test the model
* The following code will load the same configuration when your train the model:

`python3 scripts/test_agent.py --agent dqn --avg 100 --policy log/YOUR_LOG_FODLER --seed 11`

* To test on a specified map:

`python3 scripts/test_agent.py --agent dqn --avg 100 --policy log/YOUR_LOG_FODLER --seed 11 --map maps/YOUR_MAP.dat`

* Test on different agents (`prob`, `prox`, `prob_prox`, `bayes` and `optimal`): 

training config: `python3 scripts/test_agent.py --agent prob --avg 100 --policy log/YOUR_LOG_FODLER --seed 11` 

specified map: `python3 scripts/test_agent.py --agent prob --avg 100 --map maps/YOUR_MAP.dat --seed 11` 
