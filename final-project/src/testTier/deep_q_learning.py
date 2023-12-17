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
# Define the state tuple
State = namedtuple('State',
                   ('depth_image', 'relative_position'))

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

        # Seperated layers for each camera image
        self.camera_1_layer1 = nn.Linear(1024, 128)
        self.camera_1_layer2 = nn.Linear(128, 16)
        self.camera_1_layer3 = nn.Linear(16, 8)
        self.camera_2_layer1 = nn.Linear(1024, 128)
        self.camera_2_layer2 = nn.Linear(128, 16)
        self.camera_2_layer3 = nn.Linear(16, 8)
        self.camera_3_layer1 = nn.Linear(1024, 128)
        self.camera_3_layer2 = nn.Linear(128, 16)
        self.camera_3_layer3 = nn.Linear(16, 16)

        # Join the position layers
        self.joint_layer1 = nn.Linear((8 + 4 + 4), 8)

        # Join the camera layers
        self.camera_joint_layer1 = nn.Linear((8 + 8 + 16), 16)

        self.joint_layer2 = nn.Linear((8 + 16), 32)
        self.output_layer = nn.Linear(32, n_actions)

        # Debug
        self.debug = False

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state: State):
        # Feed into camera layers
        depth_im = state.depth_image
        if depth_im.dim() == 2:
            depth_im_1 = depth_im[0]
            depth_im_1 = depth_im_1.unsqueeze(0)
            depth_im_2 = depth_im[1]
            depth_im_2 = depth_im_2.unsqueeze(0)
            depth_im_3 = depth_im[2]
            depth_im_3 = depth_im_3.unsqueeze(0)
        elif depth_im.dim() == 3:
            depth_im_1 = depth_im[:,0]
            depth_im_2 = depth_im[:,1]
            depth_im_3 = depth_im[:,2]

        if self.debug:
            print("depth_im_1: ", depth_im_1)
            print("depth_im_2: ", depth_im_2)
            print("depth_im_3: ", depth_im_3)
            print("depth_im_1.dim(): ", depth_im_1.dim())
            print("depth_im_2.dim(): ", depth_im_2.dim())
            print("depth_im_3.dim(): ", depth_im_3.dim())
            print("depth_im_1.shape: ", depth_im_1.shape)
            print("depth_im_2.shape: ", depth_im_2.shape)
            print("depth_im_3.shape: ", depth_im_3.shape)

        cam1 = F.relu(self.camera_1_layer1(depth_im_1))
        cam1 = F.relu(self.camera_1_layer2(cam1))
        cam1 = F.relu(self.camera_1_layer3(cam1))
        cam2 = F.relu(self.camera_2_layer1(depth_im_2))
        cam2 = F.relu(self.camera_2_layer2(cam2))
        cam2 = F.relu(self.camera_2_layer3(cam2))
        cam3 = F.relu(self.camera_3_layer1(depth_im_3))
        cam3 = F.relu(self.camera_3_layer2(cam3))
        cam3 = F.relu(self.camera_3_layer3(cam3))

        if self.debug:
            print("cam1: ", cam1)
            print("cam2: ", cam2)
            print("cam3: ", cam3)
            print("cam1.dim(): ", cam1.dim())
            print("cam2.dim(): ", cam2.dim())
            print("cam3.dim(): ", cam3.dim())

        # Feed into position layers
        rel_pos = state.relative_position
        if rel_pos.dim() == 1:
            rel_pos = rel_pos.unsqueeze(0)
        if rel_pos.dim() == 3:
            rel_pos = rel_pos.squeeze(1)
        rel_x = rel_pos[:,0]
        rel_x = rel_x.unsqueeze(1)
        rel_y = rel_pos[:,1]
        rel_y = rel_y.unsqueeze(1)
        rel_z = rel_pos[:,2]
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
            print("x.shape: ", x.shape)
            print("y.shape: ", y.shape)
            print("z.shape: ", z.shape)

        joint_pos = torch.cat((x, y, z), dim=1)
        joint_cam = torch.cat((cam1, cam2, cam3), dim=1)
        joint_pos = F.relu(self.joint_layer1(joint_pos))
        joint_cam = F.relu(self.camera_joint_layer1(joint_cam))
        joint_all = torch.cat((joint_pos, joint_cam), dim=1)
        joint_all = F.relu(self.joint_layer2(joint_all))
        return self.output_layer(joint_all)

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

        self.num_nn_actions = 0
        self.num_random_actions = 0

    def show_action_stats(self):
        """Shows percentage of actions taken by the neural network and random actions"""
        total_actions = self.num_nn_actions + self.num_random_actions
        print(f"Total actions: {total_actions}")
        print(f"NN actions: {self.num_nn_actions} ({self.num_nn_actions/total_actions*100}%)")
        print(f"Random actions: {self.num_random_actions} ({self.num_random_actions/total_actions*100}%)")
        self.num_nn_actions = 0
        self.num_random_actions = 0

    def select_action(self, state):
        self.steps_done += 1
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
        non_final_next_states_prune = [s for s in batch.next_state if s is not None]
        non_final_next_states = State(
            depth_image=torch.stack([s.depth_image for s in non_final_next_states_prune]),
            relative_position=torch.stack([s.relative_position for s in non_final_next_states_prune])
        )
        state_batch = State(
            depth_image=torch.stack([s.depth_image for s in batch.state]),
            relative_position=torch.stack([s.relative_position for s in batch.state])
        )
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

    def test(self):
        self.gym_iface.choose_new_goal_position()
        print(f"\n\n\nDeep-QL Testing\n")
        print(f"Goal position: {self.gym_iface.goal_position}\n")
        state = State(
            depth_image=self.gym_iface.get_image_set(),
            relative_position=self.gym_iface.get_current_position().unsqueeze(0)
        )
        done_count = 0
        while done_count < 2:
            action = self.select_action(state)
            if self.debug:
                print("Selected action: ", action)
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
            # if not done:
            #     print("reward: ", reward)
            #     print("next_state: ", next_state)
            # else:
            print("reward: ", reward)

            # Store the transition in memory
            # state.relative_position = state.relative_position.unsqueeze(0)
            self.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if done:
                print("\nTesting ended due to termination or truncation\n")
                done_count += 1

        print("Done!")


    def train(self):
        num_episodes = 1000
        num_time_steps = 100
        if torch.cuda.is_available():
            num_episodes = 5000
            num_time_steps = 500

        for ep in range(num_episodes):
            if ep % 2 == 0:
                self.gym_iface.choose_new_goal_position()
            print(f"\n\n\nDeep-QL Training episode: {ep+1}\n")
            print(f"Goal position: {self.gym_iface.goal_position}\n")
            state = State(
                depth_image=self.gym_iface.get_image_set(),
                relative_position=self.gym_iface.get_current_position().unsqueeze(0)
            )
            for _ in range(num_time_steps):
                action = self.select_action(state)
                if self.debug:
                    print("Selected action: ", action)
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
                # if not done:
                #     print("reward: ", reward)
                #     print("next_state: ", next_state)
                # else:
                print("reward: ", reward)

                # Store the transition in memory
                # state.relative_position = state.relative_position.unsqueeze(0)
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
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                # Save the model - DON'T SAVE BECAUSE TESTING
                torch.save(self.policy_net.state_dict(), self.MODEL_FILE_NAME)

                if done:
                    print("\nEpisode ended due to termination or truncation\n")
                    break
            self.show_action_stats()