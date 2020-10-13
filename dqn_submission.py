import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import ShipAction, Board
from torch.utils.tensorboard import SummaryWriter

ship_actions = [
    ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH,
    ShipAction.WEST
]
BATCH_SIZE = 4
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
WINDOW_SIZE = 9

Transitions = namedtuple(
    'Transitions',
    ['state', 'action', 'next_state', 'reward']
)
writer = SummaryWriter("./runs/halite_trial_1")


class Encoder:
    def __init__(self):
        self.data = {}

    def fit(self, x):
        for idx, item in enumerate(x):
            one_hot = torch.zeros(len(x), dtype=torch.int64)
            one_hot[idx] = 1
            self.data[item] = one_hot

    def transform(self, item):
        if isinstance(item, str):
            return self.data[getattr(ShipAction, item)]
        return self.data[item]


encoder = Encoder()
encoder.fit(ship_actions)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, next_state, reward):
        if len(self) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transitions(
            state=state, action=action, next_state=next_state,
            reward=reward
        )
        self.position += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class DeepQNetwork(nn.Module):
    def __init__(self, num_actions):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.head = nn.Linear(54, num_actions)

    def forward(self, x):
        x = self.conv1(x)
        return self.head(x.view(x.size(0), -1))


def get_state(halites, ships, yards, size):
    halites = torch.tensor(halites).reshape(size, size)
    array_shape = WINDOW_SIZE * 2 + 1
    ship_pos = torch.zeros((array_shape, array_shape))
    yard_pos = torch.zeros((array_shape, array_shape))
    halites_pos = torch.zeros((array_shape, array_shape))

    for ship_id, co_ord in ships.items():
        x1, y1 = co_ord
        for x in range(WINDOW_SIZE, -WINDOW_SIZE - 1, -1):
            for y in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
                x_ = x + x1 if 0 < x + x1 < 21 else (x + x1) % 21
                y_ = y + y1 if 0 < y + y1 < 21 else (y + y1) % 21
                halites_pos[x + WINDOW_SIZE - 1][y + WINDOW_SIZE - 1] = halites[x_][y_]

        for ship_id2, co_ord2 in ships.items():
            x1_dash, y1_dash = co_ord2
            ship_x = x1 - x1_dash + WINDOW_SIZE
            ship_y = y1 - y1_dash + WINDOW_SIZE
            ship_pos[ship_x][ship_y] = 1

        for yard, yard_co_ord in yards.items():
            x1_dash, y1_dash = yard_co_ord
            yard_x = x1 - x1_dash + WINDOW_SIZE
            yard_y = y1 - y1_dash + WINDOW_SIZE
            yard_pos[yard_x][yard_y] = 1

        state = torch.stack([halites_pos, ship_pos, yard_pos])
        x, y, z = state.size()
        return state.reshape(1, x, y, z)


def agent(obs, config):
    size = config.size
    board = Board(obs, config)
    me = board.current_player

    ship_states = {}
    yard_states = {}
    exp_part = math.exp(config.steps / EPS_DECAY)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * exp_part

    sample = random.random()
    for yard in me.shipyards:
        yard_states[yard.id] = yard.position

    for ship in me.ships:
        ship_states[ship.id] = ship.position

    for ship in me.ships:
        if sample > eps_threshold:
            print("Predicting move")
            current_state = get_state(obs['halite'], ship_states, yard_states, size)
            move = policy_net(current_state).argmax().item()
            ship.next_action = ship_actions[move]
        else:
            print("random move")
            ship.next_action = random.choice(ship_actions)

    return me.next_actions, ship_states, yard_states


memory = ReplayMemory(1000)
policy_net = DeepQNetwork(len(ship_actions))
target_net = DeepQNetwork(len(ship_actions))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())


def optimize_network():
    if len(memory) > BATCH_SIZE:
        batch = memory.sample(BATCH_SIZE)
        batch = Transitions(*zip(*batch))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action).argmax(1).reshape(-1, 1)
        reward_batch = torch.tensor(batch.reward)
        next_states = torch.stack(batch.next_state)

        state_action_values = policy_net(
            state_batch).gather(1, action_batch)
        next_state_values = target_net(next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = func.smooth_l1_loss(state_action_values,
                                   expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss


def train_loop():
    env = make("halite", debug=True)
    trainer = env.train([None])
    size = env.configuration.size
    step = 0

    for idx in range(1):
        observation = trainer.reset()
        ship_states = {}
        while not env.done:
            env.configuration.steps = step
            my_action, ships, yards = agent(observation,
                                            env.configuration)
            observation = trainer.step(my_action)[0]
            halites = observation["halite"]
            halites = torch.tensor(halites).reshape(size, size)
            array_shape = WINDOW_SIZE * 2 + 1
            ship_pos = torch.zeros((array_shape, array_shape))
            yard_pos = torch.zeros((array_shape, array_shape))
            halites_pos = torch.zeros((array_shape, array_shape))
            for ship_id, co_ord in ships.items():
                x1, y1 = co_ord
                for x in range(WINDOW_SIZE, -WINDOW_SIZE - 1, -1):
                    for y in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
                        x_ = x + x1 if 0 < x + x1 < 21 else (x + x1) % 21
                        y_ = y + y1 if 0 < y + y1 < 21 else (y + y1) % 21
                        halites_pos[x + WINDOW_SIZE - 1][y + WINDOW_SIZE - 1] = halites[x_][y_]

                for ship_id2, co_ord2 in ships.items():
                    x1_dash, y1_dash = co_ord2
                    ship_x = x1 - x1_dash + WINDOW_SIZE
                    ship_y = y1 - y1_dash + WINDOW_SIZE
                    ship_pos[ship_x][ship_y] = 1

                for yard, yard_co_ord in yards.items():
                    x1_dash, y1_dash = yard_co_ord
                    yard_x = x1 - x1_dash + WINDOW_SIZE
                    yard_y = y1 - y1_dash + WINDOW_SIZE
                    yard_pos[yard_x][yard_y] = 1

                state = torch.stack([halites_pos, ship_pos, yard_pos])
                if ship_id in ship_states and len(my_action) > 0:
                    prev_state = ship_states[ship_id]
                    ship_states[ship_id] = state
                    action = encoder.transform(my_action[ship_id])
                    player = observation['player']
                    reward = observation['players'][player][0]
                    memory.push(
                        prev_state, action, state, reward
                    )
                else:
                    ship_states[ship_id] = state
            step += 1
            loss = optimize_network()
            if loss is not None:
                writer.add_scalar(
                    "training_loss",
                    loss,
                    step
                )


if __name__ == '__main__':
    train_loop()
