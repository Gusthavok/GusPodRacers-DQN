from collections import namedtuple, deque

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(n_observations, 128))
        self.dropouts.append(nn.Dropout(0.0))
        
        # Hidden layers
        for _ in range(4):
            self.layers.append(nn.Linear(128, 128))
            self.dropouts.append(nn.Dropout(0.0))
        
        # Output layer
        self.layers.append(nn.Linear(128, n_actions))

    def forward(self, x, fine_tuning=False):
        if fine_tuning:
            with torch.no_grad():
                for i in range(len(self.layers) - 1):  # Apply dropout to all but the last layer
                    x = F.relu(self.layers[i](x))
                    x = self.dropouts[i](x)
        else:
            for i in range(len(self.layers) - 1):  # Apply dropout to all but the last layer
                x = F.relu(self.layers[i](x))
                x = self.dropouts[i](x)
        # Last layer without dropout
        x = self.layers[-1](x)
        
        return x
    
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
def optimize_model(memory, policy_net, target_net, optimizer, device, GAMMA = 0.9, BATCH_SIZE = 512, fine_tuning = False):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch, fine_tuning=fine_tuning).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def select_action(state, policy_net, choose_random_action, steps_done, device, EPS_END=0.05, EPS_START=0.9, EPS_DECAY=1000):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # choose_random_action = env.action_space.sample_attaquant
        return torch.tensor([[choose_random_action()]], device=device, dtype=torch.long)
    
def soft_update(policy_net, target_net, TAU=0.005):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)
