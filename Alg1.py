import argparse
import random
import numpy as np
import gym
import nle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os
import math
import wandb
#import cv2
import gc

wandb.init(project="MuZerog17")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dtype = torch.float

### network
class ResidualBlock(nn.Module):
    def __init__(self, insize):
        super().__init__()
        self.conv1 = nn.Conv2d(insize, insize, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(insize)
        self.conv2 = nn.Conv2d(insize, insize, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(insize)
    
    def forward(self,x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.bn2(self.conv2(x1))
        x = x + x1
        return F.relu(x)

class Representation_Model(nn.Module):
    
    def __init__(self, num_hist, num_hidden):
        super().__init__()
        
        self.num_hist = num_hist
        self.num_hidden = num_hidden
        self.conv = nn.Conv2d(num_hist, 64, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.block3 = ResidualBlock(64)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.block6 = ResidualBlock(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.ln1 = nn.Linear(640, num_hidden)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.block3(x)
        x = self.pool1(x)
        x = self.block6(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)

        return self.ln1(x)

class Dynamics_Model(nn.Module):
    # action encoding - one hot
    
    def __init__(self, num_hidden, num_actions): 
        super().__init__()
        
        self.num_hidden = num_hidden
        self.num_actions = num_actions
       
        self.l1 = nn.Linear(num_hidden + 1, 128) # hidden, action encoding
        self.l2 = nn.Linear(128, 50)
        self.l5 = nn.Linear(50, num_hidden + 1) # add reward prediction
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        out = self.l5(x)
        hidden, reward = out[:, 0:self.num_hidden], out[:, -1]
        
        return hidden, reward

class Prediction_Model(nn.Module):
    
    def __init__(self, num_hidden, num_actions):
        super().__init__()
        
        self.num_actions = num_actions
        self.num_hidden = num_hidden
        
        self.l1 = nn.Linear(num_hidden, 128)
        self.network = nn.Linear(128, num_actions + 1) # value & policy prediction
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        out = self.network(x)
        p = out[:, 0:self.num_actions]
        v = out[:, -1]
        
        # softmax probs
        p = F.softmax(p, dim=1)
        return p, v


### MCTS
class MinMaxStats():
    
    def __init__(self):
        
        self.max = - np.inf
        self.min = np.inf
        
    def update(self, value):
        self.max = np.maximum(self.max, value.cpu())
        self.min = np.minimum(self.min, value.cpu())
        
    def normalize(self, value):
        value = value.cpu()
        if self.max > self.min:
            return ((value - self.min) / (self.max - self.min)).to(device)
        
        return value

class MCTS_Node():

    def __init__(self, p):
        super().__init__()
        
        self.state = None
        self.reward = None
        self.p = p
        
        self.edges = {}
        
        self.value_sum = 0
        self.visits = 0

    def expanded(self):
        return len(self.edges) > 0
        
    def search_value(self):
        if self.visits == 0:
            return 0
        return self.value_sum/self.visits

    
    
class MCTS():
    
    def __init__(self, num_actions, dynamics_model, prediction_model, agent, gamma=0.99):
        super().__init__()
        
        self.num_actions = num_actions
        self.c1 = 1.25
        self.c2 = 19652
        self.gamma = gamma
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        
        self.dynamics_model = dynamics_model
        self.prediction_model = prediction_model
        self.agent = agent
        
    def run(self, num_simulations, root_state):
        
        # init root
        p, v = self.prediction_model(root_state)
        p = p.detach()
        self.root = self.init_root(root_state, p)
        self.min_max_stats = MinMaxStats()
          
        # run simulations and save trajectory
        for i in range(num_simulations):
            
            self.node_trajectory = []
            self.node_trajectory.append(self.root)
            
            self.action_trajectory = []
            
            node = self.root
            while node.expanded():
                action, node = self.upper_confidence_bound(node)
                self.node_trajectory.append(node)
                self.action_trajectory.append(action)
                
            parent = self.node_trajectory[-2]
            v = self.expand(parent, node, self.action_trajectory[-1])

            self.backup(v)   
        return self.get_pi(), self.root.search_value()
     
    def expand(self, parent, node, action):
        
        next_state, p, v, reward = self.agent.rollout_step(parent.state, [action])
        next_state, p, v, reward = next_state.detach(), p.detach(), v.detach(), reward.detach()
        node.state = next_state
        node.reward = reward
        
        for i in range(self.num_actions):
            node.edges[i] = MCTS_Node(p[0,i])

        return v
    
    def backup(self, value):
   
        for node in reversed(self.node_trajectory):
            node.value_sum += value
            node.visits += 1
            
            self.min_max_stats.update( node.reward + self.gamma * node.search_value())
            
            value = node.reward + self.gamma * value        
            
    def upper_confidence_bound(self, node):
        ucb_scores = []
        
        for i in range(self.num_actions):
            ucb_scores.append(self.ucb_score(node,node.edges[i]))
        action = torch.argmax(torch.tensor(ucb_scores))
        x = np.argmax((torch.tensor(ucb_scores)).to('cpu').detach().numpy())
        if x == 0:
            a = ucb_scores
            maxIndex = [i for i, x in enumerate(a) if x == max(a)]
            x = random.choice(maxIndex)
            action = torch.tensor(x)
        return action, node.edges[x]
        
    def ucb_score(self, parent, child):
        
        pb_c = math.log((parent.visits + self.c2 + 1) / self.c2) + self.c1
        pb_c *= math.sqrt(parent.visits) / (child.visits + 1)
        
        prior_score = pb_c * child.p
        value_score = 0
        if child.visits > 0:
            value_score = self.min_max_stats.normalize( child.reward + self.gamma * child.search_value())
            
        return prior_score + value_score

            
    def get_pi(self):
        
        edge_visits = []
        for i in range(self.num_actions):
            edge_visits.append(self.root.edges[i].visits)
        edge_visits = np.array(edge_visits)
        
        return edge_visits
    
    def add_exploration_noise(self, node):
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * self.num_actions)

        frac = self.root_exploration_fraction
        for a, n in zip(range(self.num_actions), noise):
            node.edges[a].p = node.edges[a].p * (1 - frac) + n * frac
        return node
    
    def init_root(self, state, p):
        p = p.detach().cpu().numpy()
        
        node = MCTS_Node(0)
        node.state = state
        node.reward = 0
        
        for i in range(self.num_actions):
            node.edges[i] = MCTS_Node(p[0,i])
    
        node = self.add_exploration_noise(node)
        return node
        
    
    
### Experience_Replay
class Experience_Replay(): 
# save environment trajectories and sample sub trajectories

        
    def __init__(self, trajectory_capacity, num_actions):
        super().__init__()
        
        self.trajectory_capacity = trajectory_capacity
        self.memory = []
        self.position = 0
        self.num_actions = num_actions

    def insert(self, trajectories):
        
        for i in range(len(trajectories)):
            if len(self.memory) < self.trajectory_capacity:
                self.memory.append(None)
            self.memory[self.position] = trajectories[i]
            self.position = (self.position + 1) % self.trajectory_capacity

            
    def get_sample(self, k, n, gamma):
    # k = unroll | n = n-step-return | gamma = discount
    
        sample = {}
        sample["obs"], sample["pi"], sample["v"], sample["actions"], sample["rewards"], sample["return"] = [],[],[],[],[],[]
        
        # select trajectory
        memory_index = np.random.choice(len(self.memory),1)[0]
        traj_length = self.memory[memory_index]["length"]
        traj_last_index = traj_length - 1
        
        # select start index to unroll
        start_index = np.random.choice(traj_length, 1)[0] 
             
        # fill in the data
        sample["obs"] = self.memory[memory_index]["obs"][start_index]
        
        # compute n-step return for every unroll step, rewards and pi
        for step in range(start_index, start_index + k + 1):
        
            n_index = step + n
            
            v_n = None
            if n_index >= traj_last_index: # end of episode
                v_n = torch.tensor([0]).to(device).to(dtype)
            else:
                v_n = self.memory[memory_index]["vs"][n_index] * (gamma ** n) # discount v_n
            
            value = v_n
            # add discounted rewards until step n or end of episode
            last_valid_index = np.minimum(traj_last_index, n_index)
            for i, reward in enumerate(self.memory[memory_index]["rewards"][step:last_valid_index]):
            # rewards until end of episode
                value += reward * (gamma ** i)
                
            sample["return"].append(value)
            
            # add reward
            # only add when not inital step | dont need reward for step 0
            if step != start_index:
                if step > 0  and step <= traj_last_index:
                    sample["rewards"].append(self.memory[memory_index]["rewards"][step-1])
                else:
                    sample["rewards"].append(torch.tensor([0.0]).to(device))
                
            # add pi
            if step >= 0  and step < traj_last_index:
                sample["pi"].append(self.memory[memory_index]["pis"][step])
            else:
                sample["pi"].append(torch.tensor(np.repeat(1,self.num_actions)/self.num_actions))

        # unroll steps beyond trajectory then fill in the remaining (random) actions
        
        last_valid_index = np.minimum(traj_last_index - 1, start_index + k - 1)
        num_steps = last_valid_index - start_index
        
        # real
        sample["actions"] = self.memory[memory_index]["actions"][start_index:start_index+num_steps+1]
       
        # fills
        num_fills = k - num_steps + 1 
        for i in range(num_fills):
            sample["actions"].append(np.random.choice(self.num_actions,1)[0])
        
        return sample
        
    def get(self, batch_size, k, n, gamma=0.99):
        
        data = []
        
        for i in range(batch_size):
            sample = self.get_sample(k, n, gamma)
            data.append(sample)
            
        return data

    def __len__(self):
        return len(self.memory)
    
    

    
    
### Env_Wrapper
class Env_Wrapper(gym.Wrapper):
    # env wrapper for MuZero Cartpole, LunarLander
    
    def __init__(self, env, history_length):
        super(Env_Wrapper, self).__init__(env)
        
        self.history_length = history_length
        self.num_actions = env.action_space.n
        
    def reset(self):
    
        self.obs_history = []
        self.r = 0.0
        state = self.env.reset()
        obs1 = state["glyphs"]
        obs2 = state["blstats"]
        obs3 = state["inv_glyphs"]
        obs2 = np.concatenate([obs2[0:2],obs2[3:25],obs3])
        obs = np.concatenate([obs1,[obs2]],0)
        self.obs_history.append(obs)
        
        return self.compute_observation()
        
        
    def compute_observation(self):
        
        features = np.zeros((self.history_length, 22, 79))
        
        # features 
        current_feature_len = len(self.obs_history)

        if current_feature_len == self.history_length:
            features = np.array(self.obs_history)
        else:
            features[self.history_length-current_feature_len::] = np.array(self.obs_history)
        return features

    
    def step(self, action): 
 
        state, reward, done, info = self.env.step(action)
        obs1 = state["glyphs"]
        obs2 = state["blstats"]
        obs3 = state["inv_glyphs"]
        depth = obs2[12]
        level = obs2[18]
        # add obs and actions to history
        obs2 = np.concatenate([obs2[0:2],obs2[3:25],obs3])
        obs = np.concatenate([obs1,[obs2]],0)
        self.add_history(obs)
        self.r += reward
        if self.r <= -10.0:
            done = True
        obs = self.compute_observation()
        
        return obs, reward, done, info, depth, level
        
        
    def add_history(self, obs):
    
        if len(self.obs_history) == self.history_length:
            self.obs_history = self.obs_history[1::]
            
        self.obs_history.append(obs)

        
### Env_Runner
class Env_Runner:
    
    def __init__(self, env):
        super().__init__()
        
        self.env = env
        self.num_actions = self.env.action_space.n
        self.total_eps = 0
        
    def run(self, agent):
        
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.pis = []
        self.vs = []
        
        self.ob = self.env.reset()
        self.obs.append(torch.tensor(self.ob))
        rewards = 0.0
        done = False
        dep = 0
        lev = 0
        while not done:
            action, pi, v = agent.mcts_inference(torch.tensor(self.ob).to(device).to(dtype))
            self.ob, r, done, info, depth, level = self.env.step(action)
            self.obs.append(torch.tensor(self.ob))
            self.actions.append(action)
            self.pis.append(torch.tensor(pi))
            self.vs.append(v)
            self.rewards.append(torch.tensor(r))
            self.dones.append(done)
            rewards += r
            if dep < depth:
                dep = depth
            if lev < level:
                lev = level
        #self.env.render()
        wandb.log({"totalepi": self.total_eps,"return": rewards,"depth":dep,"level":lev})
        print(self.total_eps,rewards,dep,lev)
        self.total_eps += 1
        traject = self.make_trajectory()
        del self.obs, self.actions, self.rewards, self.dones, self.pis, self.vs
        gc.collect()
        return traject
        
        
        
    def make_trajectory(self):
        traj = {}
        traj["obs"] = self.obs
        traj["actions"] = self.actions
        traj["rewards"] = self.rewards
        traj["dones"] = self.dones
        traj["pis"] = self.pis
        traj["vs"] = self.vs
        traj["length"] = len(self.obs)
        return traj
        
        
        
def bound_state(state): # bound activations to interval [0,1]
    # probably only works when value and reward prediction are softmax over defined support ...
    
    batch_size = state.shape[0]
    
    min = torch.min(state, dim=1)[0].reshape(batch_size,1)
    max = torch.max(state, dim=1)[0].reshape(batch_size,1)
    state = (state - min) / (max - min)
    
    return state





### Agent
class MuZero_Agent(nn.Module):
    
    def __init__(self, num_simulations, num_actions, representation_model, dynamics_model, prediction_model):
        super().__init__()
        
        self.representation_model = representation_model
        self.dynamics_model = dynamics_model
        self.prediction_model = prediction_model
        
        self.num_actions = num_actions
        self.num_simulations = num_simulations
        
        self.mcts = MCTS(num_actions, dynamics_model, prediction_model, self)
        self.temperature = 1
        
    def mcts_inference(self, obs): # inference with MCTS
        obs = torch.tensor(obs[np.newaxis, :])
        start_state = self.representation_model(obs)
        child_visits, v = self.mcts.run(self.num_simulations, start_state)
        
        search_policy = child_visits/np.sum(child_visits)
        act_policy = (child_visits ** (1/self.temperature)) / np.sum(child_visits ** (1/self.temperature))
        action = np.random.choice(self.num_actions, 1, p=act_policy)
        wandb.log({"v":v,"action":action})
        return action[0], search_policy, v
  
    def inital_step(self, obs):
    # first step of rollout for optimization
        state = self.representation_model(obs)
        
        p, v = self.prediction_model(state)
        
        return state, p, v
        
        
    def rollout_step(self, state, action): 
    # unroll a step
    
        batch_size = state.shape[0]
        
        action_encoding = torch.tensor(action).to(device).to(dtype).reshape(batch_size,1) / self.num_actions
        in_dynamics = torch.cat([state,action_encoding],dim=1)
        
        next_state, reward = self.dynamics_model(in_dynamics)
        
        p, v = self.prediction_model(next_state)

        return next_state, p, v, reward
    
        
### train      
#def train():
    
history_length = 8
num_hidden = 128
num_simulations = 12
replay_capacity = 64
batch_size = 64
k = 5
n = 10
lr = 0.005
value_coef = 1#0.01#1
reward_coef = 1
policy_coef = 1
    
raw_env = gym.make('NetHackScore-v0')
num_actions = raw_env.action_space.n
    
env = Env_Wrapper(raw_env, history_length)
representation_model = Representation_Model(history_length, num_hidden).to(device)
dynamics_model = Dynamics_Model(num_hidden, num_actions).to(device)
prediction_model = Prediction_Model(num_hidden, num_actions).to(device)
    
agent = MuZero_Agent(num_simulations, num_actions, representation_model, dynamics_model, prediction_model).to(device)

runner = Env_Runner(env)
replay = Experience_Replay(replay_capacity, num_actions)
    
mse_loss = nn.MSELoss()
cross_entropy_loss = nn.CrossEntropyLoss()
logsoftmax = nn.LogSoftmax()
optimizer = optim.Adam(agent.parameters(), lr=lr)
for episode in range(4000):
    # act and get data
    trajectory = runner.run(agent)
    # save new data
    replay.insert([trajectory])
    #############
    # do update #
    #############
    if len(replay) < 15:
        continue
            
    if episode < 500:
        agent.temperature = 1
    elif episode < 1000:
        agent.temperature = 0.75
    elif episode < 1500:
        agent.temperature = 0.65
    elif episode < 2000:
        agent.temperature = 0.55
    elif episode < 3000:
        agent.temperature = 0.3
    else:
        agent.temperature = 0.25
            
    for i in range(16):
        optimizer.zero_grad()
        # get data
        data = replay.get(batch_size,k,n)

        # network unroll data
        representation_in = torch.stack([data[i]["obs"] for i in range(batch_size)]).to(device).to(dtype) 
        # flatten when insert into mem
        actions = np.stack([np.array(data[i]["actions"], dtype=np.int64) for i in range(batch_size)])
        # targets
        rewards_target = torch.stack([torch.tensor(data[i]["rewards"]) for i in range(batch_size)]).to(device).to(dtype)
        policy_target = torch.stack([torch.stack(data[i]["pi"]) for i in range(batch_size)]).to(device).to(dtype)
        value_target = torch.stack([torch.tensor(data[i]["return"]) for i in range(batch_size)]).to(device).to(dtype)
        # loss
        loss = torch.tensor(0).to(device).to(dtype)
        # agent inital step
        state, p, v = agent.inital_step(representation_in)
            
        #policy mse
        policy_loss = mse_loss(p, policy_target[:,0].detach())            
        value_loss = mse_loss(v, value_target[:,0].detach())
        loss += ( policy_coef * policy_loss + value_coef * value_loss )
        # steps
        for step in range(1, k+1):
            
            step_action = actions[:,step - 1]
            state, p, v, rewards = agent.rollout_step(state, step_action)
                
            #policy mse
            policy_loss = mse_loss(p, policy_target[:,step].detach())                
            value_loss = mse_loss(v, value_target[:,step].detach())
            reward_loss = mse_loss(rewards, rewards_target[:,step-1].detach())
            loss += ( policy_loss + value_loss + reward_loss) / k
        wandb.log({'loss': loss})
        loss.backward()
        optimizer.step() 
    model_path = 'zrepresentation1.pth'
    torch.save(representation_model.state_dict(), model_path)
    model_path = 'zdynamics1.pth'
    torch.save(dynamics_model.state_dict(), model_path)
    model_path = 'zprediction1.pth'
    torch.save(prediction_model.state_dict(), model_path)
for i in range(100):
    agent.temperature = 0.1
    trajectory = runner.run(agent)
for i in range(100):
    agent.temperature = 0.25
    trajectory = runner.run(agent)
#if __name__ == "__main__":
#
#    train()