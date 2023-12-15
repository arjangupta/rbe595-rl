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
        # Debug
        self.debug = False

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        rel_x = state[:,0]
        rel_x = rel_x.unsqueeze(1)
        rel_y = state[:,1]
        rel_y = rel_y.unsqueeze(1)
        rel_z = state[:,2]
        rel_z = rel_z.unsqueeze(1)
        if self.debug:
            print("rel_x: ", rel_x)
            print("rel_y: ", rel_y)
            print("rel_z: ", rel_z)
        x = F.relu(self.x_layer1(rel_x))
        x = F.relu(self.x_layer2(x))
        y = F.relu(self.y_layer1(rel_y))
        y = F.relu(self.y_layer2(y))
        z = F.relu(self.z_layer1(rel_z))
        z = F.relu(self.z_layer2(z))
        if self.debug:
            print("x: ", x)
            print("y: ", y)
            print("z: ", z)
            print("x.dim(): ", x.dim())
            print("y.dim(): ", y.dim())
            print("z.dim(): ", z.dim())
            # print("x.shape: ", x.shape)
            # print("y.shape: ", y.shape)
            # print("z.shape: ", z.shape)
        # if x.dim() == 2:
        #     x = x.squeeze(0)
        # if y.dim() == 2:
        #     y = y.squeeze(0)
        # if z.dim() == 2:
        #     z = z.squeeze(0)
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

        # Debug
        self.debug = False

        # Get number of actions from gym action space
        n_actions = self.gym_iface.action_primitives.NUM_ACTIONS

        # Set the model file name
        self.MODEL_FILE_NAME = "quad_dqn_model.pth"

        # Declare the policy and target networks
        n_rel_x = 1
        n_rel_y = 1
        n_rel_z = 1
        self.policy_net = QuadrotorNeuralNetwork(n_rel_x, n_rel_y, n_rel_z, n_actions).to(device)
        # If the model .pth file exists, load it
        try:
            self.policy_net.load_state_dict(torch.load(self.MODEL_FILE_NAME))
            print(f"Model file {self.MODEL_FILE_NAME} found, loading model")
        except:
            print("No model file found, creating new model")
        self.target_net = QuadrotorNeuralNetwork(n_rel_x, n_rel_y, n_rel_z, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # if self.debug:
                #     print("self.policy_net(state): ", self.policy_net(state))
                #     print("self.policy_net(state).max(0): ", self.policy_net(state).max(0))
                #     print("self.policy_net(state).max(0).indices: ", self.policy_net(state).max(0).indices)
                #     print("self.policy_net(state).max(0).indices.view(1, 1): ", self.policy_net(state).max(0).indices.view(1, 1))
                return self.policy_net(state).argmax().view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.gym_iface.action_primitives.NUM_ACTIONS)]],
                device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        if self.debug:
            print("batch: ", batch)
            print("batch.state: ", batch.state)
            print("batch.action: ", batch.action)
            print("batch.next_state: ", batch.next_state)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        if self.debug:
            print("state_batch: ", state_batch)
            print("action_batch: ", action_batch)
            print("reward_batch: ", reward_batch)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # print("action_batch: ", action_batch)
        # print("state_batch: ", state_batch)
        # print("state_action_values: ", state_action_values)
        # print("state_action_values.gather(1, action_batch): ", state_action_values.gather(1, action_batch))
        # state_action_values.gather(1, action_batch)

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
        num_episodes = 1000
        num_time_steps = 100
        if torch.cuda.is_available():
            num_episodes = 5000
            num_time_steps = 500

        for ep in range(num_episodes):
            # if ep % 5 == 0:
            print(f"Deep-QL Training episode: {ep+1}\n")
            state = self.gym_iface.get_current_position()
            for _ in range(num_time_steps):
                action = self.select_action(state)
                # observation, reward, terminated, truncated, _ =
                observation, reward, truncated, terminated = self.gym_iface.step(action.item())
                reward = torch.tensor([reward], device=device)

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    # next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    next_state = observation
                
                # if self.debug:
                if not done:
                    print("reward: ", reward)
                    print("next_state: ", next_state)

                # Store the transition in memory
                self.memory.push(state.unsqueeze(0), action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                # Save the model
                torch.save(self.policy_net.state_dict(), self.MODEL_FILE_NAME)

                if done:
                    print("\nEpisode ended due to termination or truncation\n")
                    break