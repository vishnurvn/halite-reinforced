import random
from collections import namedtuple

import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import ShipAction, Board, ShipyardAction

directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
WINDOW_SIZE = 10
MAX_SIZE = 21
actions = [
    ShipAction.NORTH,
    ShipAction.EAST,
    ShipAction.SOUTH,
    ShipAction.WEST,
    ShipAction.CONVERT,
]


transitions = namedtuple(
    'Transition',
    ['state', 'action', 'next_state', 'reward']
)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = transitions(*args)
        self.position += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def ship_collect(ship, booked_pos):
    if ship.cell.halite > 100:
        return None, None

    actual_dirs = []
    for d in directions:
        point = getattr(ship.cell, d.name.lower()).position
        if point not in booked_pos:
            actual_dirs.append((d, point))

    if len(actual_dirs) == 0:
        return None, None
    return max([d for d in actual_dirs],
               key=lambda x: getattr(ship.cell, x[0].name.lower()).halite)


def ship_deposit(ship, ship_yard, size, booked_pos):
    def get_dir_to(from_pos, to_pos, board_size):
        from_x, from_y = divmod(from_pos[0], board_size), divmod(from_pos[1], board_size)
        to_x, to_y = divmod(to_pos[0], board_size), divmod(to_pos[1], board_size)
        best_directions = []
        if from_y < to_y:
            best_directions.append(ShipAction.NORTH)
        if from_y > to_y:
            best_directions.append(ShipAction.SOUTH)
        if from_x < to_x:
            best_directions.append(ShipAction.EAST)
        if from_x > to_x:
            best_directions.append(ShipAction.WEST)

        return best_directions

    actual_dirs = []
    for d in get_dir_to(ship.position, ship_yard.position, size):
        point = getattr(ship.cell, d.name.lower()).position
        if point not in booked_pos:
            actual_dirs.append((d, point))

    if len(actual_dirs) == 0:
        return None, None
    return random.choice(actual_dirs)


def agent(obs, config):
    size = config.size
    board = Board(obs, config)
    me = board.current_player
    booked_positions = []

    if len(me.ships) < 4 and len(me.shipyards) > 0:
        yard = me.shipyards[0]
        if yard.cell.ship is None:
            yard.next_action = ShipyardAction.SPAWN

    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT

    ship_details = {}
    yard_details = {}
    ship_states = {}

    for yard in me.shipyards:
        yard_details[yard.id] = yard.position

    for ship in me.ships:
        ship_details[ship.id] = ship.position
        booked_positions.append(ship.position)
        if ship.next_action is None:
            if ship.halite < 200:
                ship_states[ship.id] = 'COLLECT'
            if ship.halite > 500:
                ship_states[ship.id] = 'DEPOSIT'

            if ship_states[ship.id] == 'COLLECT':
                next_action, booked_pt = ship_collect(ship, booked_positions)
                ship.next_action = next_action
                if booked_pt is not None:
                    booked_positions.append(booked_pt)
            if ship_states[ship.id] == 'DEPOSIT':
                next_action, booked_pt = ship_deposit(
                    ship, me.shipyards[0], size, booked_positions
                )
                ship.next_action = next_action
                if booked_pt is not None:
                    booked_positions.append(booked_pt)

    return me.next_actions, ship_details, yard_details


def make_action():
    pass


env = make("halite", debug=True)
trainer = env.train([None, 'random'])

observation = trainer.reset()

ship_states = {}
memory = ReplayMemory(1000)

while not env.done:
    my_action, ships, yards = agent(observation, env.configuration)
    observation = trainer.step(my_action)[0]
    halites = observation["halite"]
    player = observation['player']
    print(observation['players'][player][0])
    break
    # halites = np.array(halites).reshape(21, 21)
    # array_shape = WINDOW_SIZE * 2 + 1
    # ship_pos = np.zeros((array_shape, array_shape))
    # yard_pos = np.zeros((array_shape, array_shape))
    # halites_pos = np.zeros((array_shape, array_shape))
    # for ship_id, co_ord in ships.items():
    #     x1, y1 = co_ord
    #     for x in range(WINDOW_SIZE, -WINDOW_SIZE - 1, -1):
    #         for y in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
    #             x_ = x + x1 if 0 < x + x1 < 21 else (x + x1) % 21
    #             y_ = y + y1 if 0 < y + y1 < 21 else (y + y1) % 21
    #             halites_pos[x + WINDOW_SIZE - 1][y + WINDOW_SIZE - 1] = halites[x_][y_]
    #
    #     for ship_id2, co_ord2 in ships.items():
    #         x1_dash, y1_dash = co_ord2
    #         ship_x = x1 - x1_dash + WINDOW_SIZE
    #         ship_y = y1 - y1_dash + WINDOW_SIZE
    #         ship_pos[ship_x][ship_y] = 1
    #
    #     for yard, yard_co_ord in yards.items():
    #         x1_dash, y1_dash = yard_co_ord
    #         yard_x = x1 - x1_dash + WINDOW_SIZE
    #         yard_y = y1 - y1_dash + WINDOW_SIZE
    #         yard_pos[yard_x][yard_y] = 1
    #
    #     state = np.stack([halites_pos, yard_pos, ship_pos])
    #     if ship_id in ship_states:
    #         prev_state = ship_states[ship_id]
    #         ship_states[ship_id] = state
    #         memory.push(
    #             state, action, next_state, reward
    #         )
    #     else:
    #         ship_states[ship_id] = state
