import numpy as np
import worlds.craft_world as world
import copy

AgentTypes = ['Random',  'Pickup']

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

    def take_action(self, grid=None, cookbook=None):
        print('This should not be called for a neural agent')

    def reset(self):
        self.seq = []
        self.state_visit_count = 0
        self.last_states = set(self.ba.get_initial_state())

# An agent that just takes random actions
class RandomAgent(Agent):
    
    def __init__(self, init_pos, init_dir, inventory, inv_index, pred_str, is_neural=False):
        super().__init__(init_pos, init_dir, inventory, inv_index, pred_str, False)

    def take_action(self, grid=None, cookbook=None):
        action = np.random.choice(list(world.CraftWorldEnv.Actions))
        return action

# An agent that purposefully grabs items and moves them around
class PickupAgent(RandomAgent):

    def __init__(self, init_pos, init_dir, inventory, inv_index, 
                 pred_str, grid_width, grid_height, is_neural=False):
        super().__init__(init_pos, init_dir, inventory, inv_index, pred_str, False)
        
        # Keeps track of whether or not this agent is holding an item
        self.has_item = False
        
        # If an item this robot held was taken, this is true
        # This is used to prevent an agent from taking an item that has been taken
        # right back.
        self.item_taken = False

        # The probability distribution over the whole grid
        # This is used to discourage the agent from going back to the same place
        self.prob_grid = np.array([[1.] * grid_width] * grid_height)
        self.grid_width_ = grid_width
        self.grid_height_ = grid_height

        # The factor the reduce the probabilities by
        self.decay = 0.5
        
        # Maps offsets to actions
        self.del2direction = {(-1,0): int(world.CraftWorldEnv.Actions.left),
                              (1,0): int(world.CraftWorldEnv.Actions.right),
                              (0,-1): int(world.CraftWorldEnv.Actions.down),
                              (0,1): int(world.CraftWorldEnv.Actions.up),
                              (0,0): int(world.CraftWorldEnv.Actions.nothing)}

    def take_action(self, grid=None, cookbook=None):
        # If the agent had an item then doesn't, it means that another agent has taken it
        if (self.has_item and np.sum(self.get_items()) == 0):
            self.has_item = False
            self.item_taken = True

        # When the agent has an item already just move around randomly
        if (type(grid) != type(np.array([]))  or cookbook == None or self.has_item or self.item_taken):
            return super().take_action()
        
        # First find the available spaces
        spaces = set()
        for x_off in range(-1, 2):
            for y_off in range(-1, 2):
                if (x_off == 0 and y_off == 0) or (abs(x_off) == 1 and abs(y_off) == 1):
                    continue

                x = min(self.grid_width_ - 1, max(0, x_off+self.pos[0]))
                y = min(self.grid_height_ - 1, max(0, y_off+self.pos[1]))
                item = self.get_item(x, y, grid, cookbook)
                if (item == 0):
                    spaces.add((x, y, self.del2direction[x_off, y_off]))
                else:
                    # If one of the spaces is an item just pick it up
                    self.has_item = True
                    return world.CraftWorldEnv.Actions.use
        
        # Check if there are any items in the agent's line of sight
        items = cookbook.grabbable_indices
        for item in items:
            item_grid = grid[:,:,item]
            item_pos = np.array(np.where(item_grid == 1)).T
            for pos in item_pos:
                if (len(pos) == 0):
                    continue
                
                hit_item, action = self.cast_ray(pos, grid, cookbook)
                if (hit_item):
                    return action

        # Select a space based on their corresponding probabilities
        unormalized_probs = [self.prob_grid[s[0]][s[1]] for s in spaces]
        probs = np.array([p/sum(unormalized_probs) for p in unormalized_probs])
        spaces = list(spaces)
        space_indices = np.arange(len(spaces))
        space = spaces[np.random.choice(space_indices, p=probs)]

        # Reduce the probability of returning to this space
        self.prob_grid[space[0]][space[1]] *= self.decay
    
        return space[2]

    # NOTE: Will need to update if you add something else that can be moved through
    def get_item(self, x, y, grid, cookbook):
        item_indices = np.where(grid[x, y, :] == 1)
        for i in item_indices[0]:
            if not i in cookbook.color_indices:
                return i     
        return 0
    
    # Returns whether or not there exists a valid path and the first direction to move in
    def cast_ray(self, item_pos, grid, cookbook):
        vector = (np.array(item_pos) - np.array(self.pos)) 
        vector = vector/(2 * np.linalg.norm(vector))
        
        # I think NAN might be happening if the two are on top of eachother
        if not (0.4 < np.linalg.norm(vector) < 0.6):
            return (False, None)
            

        # Now use ray marching to see if there is a path between
        # the agent and the item
        current_vector = vector
        p_x, p_y = self.pos
        first_move = (0, 0)
        while True:
            ray_pos = np.round(self.pos + current_vector)
            x, y = [int(np.round(ray_pos[i])) for i in range(2)]
            dx = x - int(p_x)
            dy = y - int(p_y)
            
            # This ensures that the first move is not within the same square
            # and that it is not a diagonal move
            if (first_move == (0, 0)):
                if (abs(dx) == 1 and abs(dy) == 1):
                    if (abs(vector[0]) > abs(vector[1])):
                        first_move = (dx, 0)
                    else:
                        first_move = (0, dy)
                else:
                    first_move = (dx, dy)
            
            if (x == item_pos[0] and y == item_pos[1]):
                return (True, self.del2direction[first_move])

            if (x >= self.grid_width_ or x < 0 or y >= self.grid_height_ or y < 0):
                return (False, None)

            # I check two items to ensure that the path is clear along the diagonal
            # if the ray moves diagonally
            item_1 = self.get_item(x, y, grid, cookbook)
            item_2 = self.get_item(x - dx, y - dy, grid, cookbook)
            if (item_1 != 0 or item_2 != 0):
                return (False, None)
            
            # Update the previous positions and the current vector
            p_x, p_y = (x, y)
            current_vector += vector


