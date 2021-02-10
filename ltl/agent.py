import numpy as np
import worlds.craft_world as world
import copy

class Agent:
    
    def __init__(self, init_pos, init_dir, inventory, inv_index, pred_str, is_neural=False):
        self._init_pos = init_pos
        self._init_dir = init_dir
        self.is_neural = is_neural
        self.inventory = inventory

        # The row of corresponding to the index of this agent in the inventory matrix
        self.inv_index = inv_index
        
        self.pred_str = pred_str
        self.pos = copy.deepcopy(init_pos)
        self.dir = copy.deepcopy(init_dir)
    
    def take_action(self, grid=[], cookbook={}):
        pass

    def get_items(self):
        return self.inventory
    
    def add_items(self, item_index, count=1):
        self.inventory[item_index] += count
    
    def remove_items(self, item_index, count=1):
        self.inventory[item_index] = max(0, self.inventory[item_index] - count)
    
    def clear_inventory(self):
        self.inventory[:] = 0

    def reset(self):
        self.pos = copy.deepcopy(self._init_pos)
        self.dir = copy.deepcopy(self._init_dir)
        self.clear_inventory()

# An agent to be used as a container for agents driven by a neural network
class NeuralAgent(Agent):
    
    def __init__(self, init_pos, init_dir, inventory, inv_index, pred_str, formula, ba):
        super().__init__(init_pos, init_dir, inventory, inv_index, pred_str, True)
        self.formula = formula
        self.ba = ba

        self.seq = []
        self.state_visit_count = 0
        self.last_states = set(ba.get_initial_state())

    def take_action(self, grid=[], cookbook={}):
        print('This should not be called for a neural agent')

    def reset(self):
        super().reset()
        self.seq = []
        self.state_visit_count = 0
        self.last_states = set(self.ba.get_initial_state())

# An agent that just takes random actions
class RandomAgent(Agent):
    
    def __init__(self, init_pos, init_dir, inventory, inv_index, pred_str, is_neural=False):
        super().__init__(init_pos, init_dir, inventory, inv_index, pred_str, False)

    def take_action(self, grid=[], cookbook={}):
        action = np.random.choice(list(world.CraftWorldEnv.Actions))
        return action

