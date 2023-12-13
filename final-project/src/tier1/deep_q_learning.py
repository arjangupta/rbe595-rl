import math
import random
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """Experience Replay memory"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class QuadrotorNeuralNetwork(nn.Module):

    def __init__(self, n_rel_x, n_rel_y, n_rel_z, n_actions):
        super(QuadrotorNeuralNetwork, self).__init__()
        # Separated layers for each relative position
        self.x_layer1 = nn.Linear(n_rel_x, 8)
        self.x_layer2 = nn.Linear(8, 8)
        self.y_layer1 = nn.Linear(n_rel_y, 8)
        self.y_layer2 = nn.Linear(8, 4)
        self.z_layer1 = nn.Linear(n_rel_z, 8)
        self.z_layer2 = nn.Linear(8, 4)
        # Join the layers
        self.joint_layer1 = nn.Linear((8 + 4 + 4), 8)
        self.joint_layer2 = nn.Linear(8, 32)
        self.output_layer = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, rel_x, rel_y, rel_z):
        x = F.relu(self.x_layer1(rel_x))
        x = F.relu(self.x_layer2(x))
        y = F.relu(self.y_layer1(rel_y))
        y = F.relu(self.y_layer2(y))
        z = F.relu(self.z_layer1(rel_z))
        z = F.relu(self.z_layer2(z))
        joint = torch.cat((x, y, z), dim=1)
        joint = F.relu(self.joint_layer1(joint))
        joint = F.relu(self.joint_layer2(joint))
        return self.output_layer(joint)

class DeepQLearningAgent:
    def __init__(self, gym_iface):
        self.gym_iface = gym_iface
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer
        self.BATCH_SIZE = 64
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4

        # Get number of actions from gym action space
        n_actions = self.gym_iface.action_primitives.NUM_ACTIONS

        # Declare the policy and target networks
        n_rel_x = 1
        n_rel_y = 1
        n_rel_z = 1
        self.policy_net = QuadrotorNeuralNetwork(n_rel_x, n_rel_y, n_rel_z, n_actions).to(device)
        self.target_net = QuadrotorNeuralNetwork(n_rel_x, n_rel_y, n_rel_z, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0


    def select_action(self, rel_x, rel_y, rel_z):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(rel_x, rel_y, rel_z).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.gym_iface.action_primitives.NUM_ACTIONS)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
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
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        if torch.cuda.is_available():
            num_episodes = 5000
        else:
            num_episodes = 1000

        for ep in range(num_episodes):
            state = self.gym_iface.get_current_position()
            # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            curr_pos_x = state[0], curr_pos_y = state[1], curr_pos_z = state[2]
            for t in count():
                action = self.select_action(curr_pos_x, curr_pos_y, curr_pos_z)
                # observation, reward, terminated, truncated, _ =
                next_state, reward = self.gym_iface.step(action.item())
                reward = torch.tensor([reward], device=device)
                # done = terminated or truncated

                # if terminated:
                #     next_state = None
                # else:
                #     next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)