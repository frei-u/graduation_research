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
import os
import math
#import wandb
#import cv2
import gc

#wandb.init(project="StoMuZero")

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
    
    def __init__(self, num_hist):
        super().__init__()
        
        self.num_hist = num_hist
        self.conv = nn.Conv2d(num_hist, 64, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.block3(x)
        x = self.block4(x)
        
        return self.pool1(x)

class Dynamics_Model(nn.Module):
    # action encoding - one hot
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(64+32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.block1 = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.rconv = nn.Conv2d(64+32, 1, kernel_size=1, stride=1, padding=1)
        self.rbn = nn.BatchNorm2d(1)
        self.ln = nn.Linear(60,1)
    
    def forward(self, x, c, batch):
        ac = c.repeat_interleave(3*10,dim=1)
        ac = torch.reshape(ac, (batch, 32, 3, 10))
        x = torch.cat([x,ac],dim=1)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.block1(x1)
        
        r = F.relu(self.rbn(self.rconv(x)))
        r = torch.flatten(r,1)
        return F.relu(self.bn3(self.conv3(x1))), self.ln(r)
    
class Prediction_Model(nn.Module):
    
    def __init__(self, num_actions):
        super().__init__()
        
        self.conv1 = nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.network = nn.Linear(240, num_actions)
        
        self.vconv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=1)
        self.vbn = nn.BatchNorm2d(1)
        self.ln = nn.Linear(60,1)
    
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = torch.flatten(x1, 1)
        p = self.network(x1)
        v = F.relu(self.vbn(self.vconv(x)))
        v = torch.flatten(v, 1)
        # softmax probs
        return F.softmax(p, dim=1), self.ln(v)
    
class AfterstateDynamics_Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64+23, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.block1 = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
    
    def forward(self, x, action, batch):
        ac = F.one_hot(action,23)
        ac = ac.repeat_interleave(3*10,dim=1)
        ac = torch.reshape(ac, (batch, 23, 3, 10))
        x = torch.cat([x,ac],dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        
        return F.relu(self.bn3(self.conv3(x)))
    
class AfterstatePrediction_Model(nn.Module):
    
    def __init__(self, csize):
        super().__init__()
        
        self.conv1 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.network = nn.Linear(120, csize)
        
        self.vconv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=1)
        self.vbn = nn.BatchNorm2d(1)
        self.ln = nn.Linear(60,1)
    
    def forward(self, x):
        code = F.relu(self.bn1(self.conv1(x)))
        code = torch.flatten(code, 1)
        code = self.network(code)
        
        q = F.relu(self.vbn(self.vconv(x)))
        q = torch.flatten(q, 1)
        # softmax probs
        return F.softmax(code, dim=1), self.ln(q)
    

class VQ_VAE(nn.Module):
    
    def __init__(self, num_hist):
        super().__init__()
        self.size = 32
        self.codebook = torch.tensor(torch.eye(self.size)[np.newaxis,:]).to(device)
        
        self.conv1 = nn.Conv2d(num_hist, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.ln = nn.Linear(160,32)
        
    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.ln(x)
        
        return self.encode(x), x
    
    def encode(self, x):
        batch = 64
        codebook = (self.codebook).repeat_interleave(batch, dim=0)
        x = x.repeat_interleave(self.size, dim=1)
        x = torch.reshape(x, (-1, self.size, self.size))
        distance = torch.sum((x - codebook) ** 2, dim=2)
        indices = torch.argmin(distance, dim=1)
        onehot = torch.eye(self.size)[indices]
        onehot = onehot.repeat_interleave(self.size, dim=1).reshape((-1, self.size, self.size)).to(device)  # [b, c, c]
        
        return torch.sum(onehot * codebook, dim=2)  # [b, c, c]->[b, c]
    

### MCTS
   
class MCTS_Node():

    def __init__(self, p, is_chance = False):
        super().__init__()
        
        self.reward = 0
        self.p = p
        self.is_chance = False
        self.edges = {}
        self.value_sum = 0
        self.is_chance = is_chance
        self.visits = 0

    def expanded(self):
        return len(self.edges) > 0
        
    def search_value(self):
        if self.visits == 0:
            return 0
        return self.value_sum/self.visits


    
    
class MCTS():
    
    def __init__(self, num_actions, agent, gamma=0.99):
        super().__init__()
        
        self.num_actions = num_actions
        self.c1 = 1.25
        self.c2 = 19652
        self.gamma = gamma
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.agent = agent
        
    def run(self, num_simulations, root_state):
        
        # init root
        p, v = self.agent.f(root_state)
        self.root = self.init_root(p[0].detach().cpu().numpy())
        
        # run simulations and save trajectory
        for _ in range(num_simulations):
            state = root_state
            self.node_trajectory = [self.root]
            node = self.root
            self.action_trajectory = []
            while node.expanded():
                action, node = self.upper_confidence_bound(node)
                self.node_trajectory.append(node)
                self.action_trajectory.append(action)
            parent = self.node_trajectory[-2]
            
            if parent.is_chance:
                state, r = self.agent.g(state, F.one_hot(torch.tensor(action).to(device),32)[np.newaxis, :])
                p, v = self.agent.f(state)
                is_child_chance = False
                a = self.num_actions
            else:
                state = self.agent.phi(state, torch.tensor([action]).to(device))
                p, v = self.agent.psi(state) # p = sigma
                a = 32
                r = 0.0
                is_child_chance = True
            
            self.expand(node, p, r, is_child_chance, a)
            self.backup(v)      

        return self.get_pi(), self.root.search_value()
     
    def expand(self, node, p, r, is_chance, n):
        p = p[0].detach().cpu().numpy()#, r.detach()
        node.reward = r
        node.is_chance = is_chance
        for i in range(n):
            node.edges[i] = MCTS_Node(p[i])
    
    def backup(self, value):
        #value = value.detach()
        for node in reversed(self.node_trajectory):
            node.value_sum += value
            node.visits += 1
            value = node.reward + self.gamma * value

    def get_pi(self):
        
        edge_visits = []
        for i in range(self.num_actions):
            edge_visits.append(self.root.edges[i].visits)
        edge_visits = np.array(edge_visits)
        
        return edge_visits
            
    def upper_confidence_bound(self, node):
        ucb_scores = []
        if node.is_chance:
            for i in range(32):
                ucb_scores.append(node.edges[i].p)
            ucb_scores = ucb_scores/sum(ucb_scores)
            outcome = np.random.choice(32,p=ucb_scores)
            return torch.tensor(outcome), node.edges[outcome]
                
        for i in range(self.num_actions):
            ucb_scores.append(self.ucb_score(node,node.edges[i]))
        action = torch.argmax(torch.tensor(ucb_scores))
        x = np.argmax((torch.tensor(ucb_scores)).to('cpu').detach().numpy())
        if x == 0:
            maxIndex = [i for i, x in enumerate(ucb_scores) if x == max(ucb_scores)]
            x = random.choice(maxIndex)
            action = torch.tensor(x)
        return action, node.edges[x]
        
    def ucb_score(self, parent, child):
        
        prior_score = (math.log((parent.visits + self.c2 + 1) / self.c2) + self.c1) * math.sqrt(parent.visits) / (child.visits + 1) * child.p
        if child.visits > 0:
            return prior_score + child.reward + self.gamma * child.search_value()
            
        return prior_score

        
    def init_root(self, p):
        
        node = MCTS_Node(0)
        node.reward = 0
        #add_exploration_noise
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * self.num_actions)
        p = np.add(p * (1 - self.root_exploration_fraction), noise * self.root_exploration_fraction)
        for i in range(self.num_actions):
            node.edges[i] = MCTS_Node(p[i])
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
            if step > start_index and step <= traj_last_index:
                sample["rewards"].append(self.memory[memory_index]["rewards"][step-1])
            else:
                sample["rewards"].append(torch.tensor([0.0]).to(device))
                
            # add pi
            if step >= 0 and step < traj_last_index:
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
        for _ in range(batch_size):
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
        self.r = 0.0
        self.d = 1
        state = self.env.reset()
        obs1 = state["glyphs"]
        obs2 = state["blstats"]
        obs3 = state["inv_glyphs"]
        obs = np.concatenate([obs2[0:2],obs2[3:25],obs3])
        obs = np.concatenate([obs1,[obs]],0)
        self.obs_history = [obs]
        self.nothingobs = np.count_nonzero(obs1 == 2359)
        self.l = obs2[18]
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
        # add obs and actions to history
        obs = np.concatenate([obs2[0:2],obs2[3:25],obs3])
        obs = np.concatenate([obs1,[obs]],0)
        self.add_history(obs)
        self.r += reward
        
        obs = self.compute_observation()
        add = 0
        if action == 0:
            add = -0.5
        a = np.count_nonzero(obs1 == 2359)
        if not done:
            if obs2[12] > self.d:
                add += obs2[12]*200
                self.d = obs2[12]
            elif self.nothingobs > a:
                add += 0.01*(self.nothingobs-a)
            if obs2[21] == 1:
                add += 0.05
            elif obs2[21] == 0:
                add -= 0.02
            else:
                add -= (obs2[21]-1)*0.1
            if self.l < obs2[18]:
                add += 50*(obs2[18]-self.l)
                self.l = obs2[18]
            add += min(obs2[10]/obs2[11]-0.25,0.25)/50
        
        self.nothingobs = a
        return obs, reward, done, info, add, self.d, self.l
        
        
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
        allrewards = 0
        dep = 0
        lev = 0
        done = False
        #self.env.render()
        while not done:
            action, pi, v = agent.mcts_inference(torch.tensor(self.ob).to(device).to(dtype))
            self.ob, r, done, info, addp, depth, level = self.env.step(action)
            rewards += r
            allrewards += r+addp
            if dep < depth:
                dep = depth
            if lev < level:
                lev = level
            if allrewards <= -100.0 or rewards <= -10.0:
                done = True
            self.obs.append(torch.tensor(self.ob))
            self.actions.append(action)
            self.pis.append(torch.tensor(pi))
            self.vs.append(v)
            self.rewards.append(torch.tensor(r+addp))
            self.dones.append(done)
        #self.env.render()
        #wandb.log({"totalepi": self.total_eps,"return": rewards,"return+": allrewards, "depth":dep, "level":lev})
        print(self.total_eps,rewards,"\t\t\t\t\t",dep,"\t\t\t\t\t",allrewards,"\t\t\t\t\t",lev)
        self.total_eps += 1
        return self.make_trajectory()
        
        
        
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
    
    def __init__(self, num_simulations, num_actions, representation_model, dynamics_model, prediction_model, afterstatedynamics_model, afterstateprediction_model, vq_vae):
        super().__init__()
        
        self.representation_model = representation_model
        self.dynamics_model = dynamics_model
        self.prediction_model = prediction_model
        self.afterstatedynamics = afterstatedynamics_model
        self.afterstateprediction = afterstateprediction_model
        self.vq_vae = vq_vae
        
        self.num_actions = num_actions
        self.num_simulations = num_simulations
        self.temperature = 1
        self.mcts = MCTS(num_actions, self)
        
    def mcts_inference(self, obs): # inference with MCTS
        obs = torch.tensor(obs[np.newaxis, :])
        start_state = self.h(obs)
        child_visits, v = self.mcts.run(self.num_simulations, start_state)
        
        search_policy = child_visits/np.sum(child_visits)
        act_policy = (child_visits ** (1/self.temperature)) / np.sum(child_visits ** (1/self.temperature))
        action = np.random.choice(self.num_actions, 1, p=act_policy)
        #wandb.log({"v":v,"action":action})
        return action[0], search_policy, v
  
    def h(self,obs):
        #retrn state 
        return self.representation_model(obs)

    def vae(self,obs):
        #return code
        return self.vq_vae(obs)
    
    def f(self,state):
        # return p, v
        return self.prediction_model(state)
    
    def phi(self, state, action):
        # return after_state
        return self.afterstatedynamics(state,torch.tensor(action).to(device),state.shape[0])

    def psi(self, after_state):
        # return chance and Q(Q:value)
        return self.afterstateprediction(after_state)
    
    def g(self, after_state, chance):
        # return state, reward        
        return self.dynamics_model(after_state,torch.tensor(chance).to(device),after_state.shape[0])
    
        
### train      
def train():
    
    history_length = 8
    num_simulations = 50
    replay_capacity = 96
    batch_size = 64
    k = 5
    n = 10
    lr = 0.01
    
    raw_env = gym.make('NetHackScore-v0')
    num_actions = raw_env.action_space.n
    #torch.nn.DataParallel(net, device_ids=[0, 1])
    env = Env_Wrapper(raw_env, history_length)
    representation_model = Representation_Model(history_length).to(device)
    dynamics_model = Dynamics_Model().to(device)
    prediction_model = Prediction_Model(num_actions).to(device)
    afterstatedynamics_model = AfterstateDynamics_Model().to(device)
    afterstateprediction_model = AfterstatePrediction_Model(32).to(device)
    vq_vae = VQ_VAE(history_length).to(device)
    
    agent = MuZero_Agent(num_simulations, num_actions, representation_model, dynamics_model, prediction_model, afterstatedynamics_model, afterstateprediction_model, vq_vae).to(device)

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
            lr = 0.01
        elif episode < 1000:
            agent.temperature = 0.8
            lr = 0.008
        elif episode < 1500:
            agent.temperature = 0.5
            lr = 0.006
        elif episode < 2000:
            agent.temperature = 0.4
            lr = 0.005
        elif episode < 3000:
            agent.temperature = 0.3
            lr = 0.003
        else:
            agent.temperature = 0.25
            lr = 0.001
        if episode == 500 or episode == 1000 or episode == 1500 or episode == 2000 or episode == 3000:
            optimizer = optim.Adam(agent.parameters(), lr=lr)    
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
            state = agent.h(representation_in)
            p, v = agent.f(state)

            v = torch.flatten(v)
            code, bcode = agent.vae(representation_in)
            c = torch.tensor(code[np.newaxis, :])
            policy_loss = 100*mse_loss(p, policy_target[:,0].detach())            
            value_loss = 0.01*mse_loss(v, value_target[:,0].detach())
            vq_loss = 0
            reward_loss = 0
            q_loss = 0
            sigma_loss = 0
            # steps
            for step in range(1, k+1):
                step_action = actions[:,step - 1]
                after_state = agent.phi(state, step_action)
                sigma, q = agent.psi(after_state)
                ob = representation_in[step]
                ob = torch.tensor(ob[np.newaxis, :])
                state, rewards = agent.g(after_state, c)
                p, v = agent.f(state)

                v = torch.flatten(v)
                rewards = torch.flatten(rewards)
                q = torch.flatten(q)
                policy_loss += 100*mse_loss(p, policy_target[:,step].detach())                
                value_loss += 0.01*mse_loss(v, value_target[:,step].detach())
                reward_loss += 0.05*mse_loss(rewards, rewards_target[:,step-1].detach())
                q_loss += 0.01*mse_loss(q, value_target[:,step-1].detach())
                sigma_loss += 20*mse_loss(sigma, code.detach())
                vq_loss += min(20*mse_loss(code[step-1], bcode[step-1]),25)
            loss = policy_loss + value_loss + reward_loss + q_loss + sigma_loss + vq_loss
            #wandb.log({'policy loss': policy_loss, 'value loss': value_loss, 'reward loss': reward_loss, 'q loss': q_loss, 'sigma loss':sigma_loss, 'vq loss': vq_loss, 'loss': loss})
            loss.backward()
            optimizer.step() 
        model_path = 'representation2.pth'
        torch.save(representation_model.state_dict(), model_path)
        model_path = 'dynamics2.pth'
        torch.save(dynamics_model.state_dict(), model_path)
        model_path = 'prediction2.pth'
        torch.save(prediction_model.state_dict(), model_path)
        model_path = 'afterstatedynamics2.pth'
        torch.save(afterstatedynamics_model.state_dict(), model_path)
        model_path = 'afterstateprediction2.pth'
        torch.save(afterstateprediction_model.state_dict(), model_path)
        model_path = 'vq_vae2.pth'
        torch.save(vq_vae .state_dict(), model_path)
    print("eval")
    for episode in range(100):
        # act and get data
        agent.temperature = 0.1
        trajectory = runner.run(agent)        
if __name__ == "__main__":

    train()