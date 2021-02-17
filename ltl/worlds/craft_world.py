from enum import IntEnum
from gym import spaces
from ltl.ltl2tree import *
from PIL import Image
from skimage.measure import block_reduce
from skimage.transform import resize
from skimage.util import pad
from ltl.spot2ba import Automaton
from ltl.utils import Index, pad_slice
from pathlib import Path

import ltl.agent as Agent
import itertools
import argparse
import copy
import gym
import numpy as np
import os
import pygame
import time
import yaml
import json
import pickle

# ROOT_PATH = "/Users/dsleeps/Documents/ltl-environment-dev/ltl/worlds/" #TODO adjust this
ROOT_PATH = "/storage/dsleeper/RL_Parser/ltl/ltl/worlds/"

def neighbors(pos, width, height, dir=None):
    x, y = pos
    neighbors = []
    if x > 0 and (dir is None or dir == CraftWorldEnv.Actions.left):
        neighbors.append((x-1, y))
    if y > 0 and (dir is None or dir == CraftWorldEnv.Actions.down):
        neighbors.append((x, y-1))
    if x < width - 1 and (dir is None or dir == CraftWorldEnv.Actions.right):
        neighbors.append((x+1, y))
    if y < height - 1 and (dir is None or dir == CraftWorldEnv.Actions.up):
        neighbors.append((x, y+1))
    return neighbors


def nears(pos, width, height):
    x, y = pos
    neighbors = []
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            nx = x + dx
            ny = y + dy
            if nx == x and ny == y:
                continue
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            neighbors.append((nx, ny))
    return neighbors

def remap_recipes(recipes):
    '''
    gem -> apple
    gold -> orange
    iron -> pear

    factory -> flag
    tree -> tree
    workbench -> house
    '''
    fruit_recipes = recipes.copy()
    fruit_recipes["primitives"].remove("iron") 
    fruit_recipes["primitives"].remove("gem") 
    fruit_recipes["primitives"].remove("gold") 
    
    fruit_recipes["primitives"] += ["orange", "apple", "pear"]

    fruit_recipes["environment"].remove("recycle_gem") 
    fruit_recipes["environment"].remove("recycle_gold") 
    fruit_recipes["environment"].remove("recycle_iron") 

    fruit_recipes["environment"] += ["orange_bin", "apple_bin", "pear_bin"]
    
    fruit_recipes["environment"].remove("factory") 
    fruit_recipes["environment"].remove("workbench") 

    fruit_recipes["environment"] += ["flag", "house"]

    for x,y in zip(["none0", "none2", "none3"], ["pear", "orange", "apple"]):
        popx = fruit_recipes["recipes"].pop(x)
        popx["_at"] = y+"_bin"
        fruit_recipes["recipes"][y+"_none"] = popx

    return fruit_recipes

class Cookbook(object):
    def __init__(self, recipes_path):
        with open(recipes_path) as recipes_f:
            recipes = yaml.load(recipes_f, Loader=yaml.FullLoader)
            # print(recipes)
        #if "fruit" not in recipes_path:
        #    recipes = remap_recipes(recipes)
        self.index = Index()
        self.environment = set(self.get_index(e) for e in recipes["environment"])
        self.primitives = set(self.get_index(p) for p in recipes["primitives"])
        # The word after the underscore is the name of the agent
        self.det_agents = [self.get_index(det_agent.split("_")[1]) for det_agent in recipes["det_agents"]]
        self.det_agent_types = [det_agent.split("_")[0] for det_agent in recipes["det_agents"]]
        self.det_agent_names = [det_agent.split("_")[1] for det_agent in recipes["det_agents"]]

        self.recipes = {}
        self.original_recipes = recipes
        self.output2input = {} #maps none0, none2, none3 to input items
        # set up recipes from yaml file
        if 'recipes' in recipes.keys():
            for output, inputs in recipes["recipes"].items():
                #print(output)
                #print(inputs)
                #exit()
                d = {}
                for inp, count in inputs.items():
                    if "_" in inp:  # special keys
                        d[inp] = count
                    else:
                        d[self.get_index(inp)] = count
                        self.output2input[output] = inp#assumes there is only one input
                self.recipes[self.get_index(output)] = d
        self.input2output = {x:y for y,x in self.output2input.items()} 
        kinds = self.environment | self.primitives | set(self.recipes.keys())
        self.n_kinds = len(self.index)
        # get indices
        self.grabbable_indices = [i+1 for i in range(self.n_kinds)
                                  if i+1 not in self.environment]
        self.workshop_indices = [item for item in self.environment if 'recycle' in self.index.get(item)]
        self.out_indices = [item for item in self.environment if 'none' in self.index.get(item)]#outputs of workshop
        self.switch_indices = [item for item in self.environment if 'switch' in self.index.get(item)]
        self.color_indices = [item for item in self.environment if 'color_swap' in self.index.get(item)]
        self.water_index = self.index["water"]
        self.stone_index = self.index["stone"]

    def primitives_for(self, goal):
        out = {}

        def insert(kind, count):
            assert kind in self.primitives
            if kind not in out:
                out[kind] = count
            else:
                out[kind] += count

        for ingredient, count in self.recipes[goal].items():
            if not isinstance(ingredient, int):
                assert ingredient[0] == "_"
                continue
            elif ingredient in self.primitives:
                insert(ingredient, count)
            else:
                sub_recipe = self.recipes[ingredient]
                n_needed = count
                expanded = self.primitives_for(ingredient)
                for k, v in expanded.items():
                    insert(k, v * n_needed)
        return out

    def get_index(self, item):
        return self.index.index(item)

    def __str__(self):
        out_str = ''
        for item in self.index:
            out_str = '{}{}\t{}\n'.format(out_str, item, self.get_index(item))
        return out_str


class CraftGui(object):
    def __init__(self, env, width, height, is_headless=False,
                 width_px=400, height_px=400, target_fps=None,
                 img_path=ROOT_PATH+'/fruit_images/',
                 caption='CraftWorld Simulator'):
        if is_headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self._env = env
        self._width = width
        self._height = height
        self._cell_width = width_px / width
        self._cell_height = height_px / height
        # load icon images
        self._sprites = {}
        for item in env.cookbook.index:
            if 'none' in item:
                continue
            if item == 'boundary':
                for direction in ['left', 'right', 'top', 'bottom']:
                    filename = img_path + item + '_' + direction + '.png'
                    self._sprites[item  + '_' + direction] = \
                        pygame.image.load(filename)
            else:
                if 'recycle' in item:
                    bin_name = item.split('_')[1] + "_bin"
                    self._sprites[bin_name+"_full"] = pygame.image.load(img_path + bin_name + '_full.png')
                    self._sprites[bin_name+"_empty"] = pygame.image.load(img_path + bin_name + '_empty.png')
                else:
                    self._sprites[item] = pygame.image.load(img_path + item + '.png')

        # Load the agent images
        for direction in ['left', 'right', 'top', 'bottom']:
            self._sprites[direction + '_empty'] = pygame.image.load(img_path + direction + '_empty.png')

        # pygame related
        self._target_fps = target_fps
        self._screen = pygame.display.set_mode((width_px, height_px), 0, 32)
        pygame.display.set_caption(caption)
        self._clock = pygame.time.Clock()

    def move(self, move_idx=None):
        if move_idx is not None:
            obs, reward, done, _ = self._env.step(move_idx)
            return
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    print(self._env.actions.up)
                    obs, reward, done, _ = self._env.step([self._env.actions.up])
                elif event.key == pygame.K_DOWN:
                    print(self._env.actions.down)
                    obs, reward, done, _ = self._env.step([self._env.actions.down])
                elif event.key == pygame.K_LEFT:
                    print(self._env.actions.left)
                    obs, reward, done, _ = self._env.step([self._env.actions.left])
                elif event.key == pygame.K_RIGHT:
                    print(self._env.actions.right)
                    obs, reward, done, _ = self._env.step([self._env.actions.right])
                elif event.key == pygame.K_SPACE:
                    #print("about to step*******")
                    print(self._env.actions.use)
                    obs, reward, done, _ = self._env.step([self._env.actions.use])
                    #print("use")
                    self._env.print_inventory()
                    #exit()
                else:
                    continue
                print('reward: {}, done: {}'.format(reward, done))

    def draw(self, move_idx=None, move_first=False):
        if move_first:
            self.move(move_idx=move_idx)
        bg_color = (255, 255, 255)
        #bg_color = (0, 0, 0)
        self._screen.fill(bg_color)
        row = 0
        cell_size = (int(self._cell_width-1)-2, int(self._cell_height-1)-2)
        for y in reversed(range(self._height)):
            for x in range(self._width):
                px_x = x*self._cell_width
                px_y = row*self._cell_height
                rect = pygame.Rect(px_x, px_y, self._cell_width, self._cell_height)
                pygame.draw.rect(self._screen, (00,00,00), rect, 2)
                if self._env.get_item(x, y) or (x, y) == self._env.pos:
                    thing = self._env.get_item(x, y)
                    for agent in self._env.all_agents:
                        if (x, y) == agent.pos:
                            if (agent.dir == CraftWorldEnv.Actions.left):
                                picture = pygame.transform.scale(self._sprites['left_empty'], cell_size)
                            if (agent.dir == CraftWorldEnv.Actions.right):
                                picture = pygame.transform.scale(self._sprites['right_empty'], cell_size)
                            if (agent.dir == CraftWorldEnv.Actions.up):
                                picture = pygame.transform.scale(self._sprites['up_empty'], cell_size)
                            else:
                                picture = pygame.transform.scale(self._sprites['down_empty'], cell_size)
                    if thing == self._env.cookbook.get_index("boundary"):
                        if row == 0:
                            picture = pygame.transform.scale(self._sprites['boundary_top'], cell_size)
                        elif row == self._height - 1:
                            picture = pygame.transform.scale(self._sprites['boundary_bottom'], cell_size)
                        elif x == 0:
                            picture = pygame.transform.scale(self._sprites['boundary_left'], cell_size)
                        elif x == self._width - 1:
                            picture = pygame.transform.scale(self._sprites['boundary_right'], cell_size)
                    elif "recycle" in self._env.cookbook.index.get(thing):
                        for output, inputs in self._env.cookbook.recipes.items():
                            if inputs["_at"] != self._env.cookbook.index.get(thing):
                                continue
                            recycle_name = self._env.cookbook.index.get(thing)
                            bin_name = recycle_name.split('_')[1] + "_bin"
                            if self._env.workshop_outs[output] > 0:
                                picture = pygame.transform.scale(self._sprites[bin_name + "_full"],
                                                                 cell_size)
                            else:
                                picture = pygame.transform.scale(self._sprites[bin_name + "_empty"],
                                                                 cell_size)
                    else:
                        picture = pygame.transform.scale(self._sprites[self._env.cookbook.index.get(thing)],
                                                         cell_size)
                    self._screen.blit(picture, (px_x+2, px_y+2))
            row += 1
        pygame.display.update()
        if self._target_fps is not None:
            self._clock.tick(self._target_fps)


class CraftWorldEnv(gym.Env):
    class Actions(IntEnum):
        down  = 0 # move down
        up    = 1 # move up
        left  = 2 # move left
        right = 3 # move right
        nothing=4 # do nothing
        use   = 5 # use

    def __init__(self, formulas, recipe_path,
                 init_pos, init_dir, grid, neural_agent_num=1,
                 width=7, height=7, window_width=5, window_height=5,
                 prefix_reward_decay=1., time_limit=10,
                 use_gui=True, target_fps=None, is_headless=False,
                 update_failed_trans_only=False):
        self.first_accept = 100
        self.cookbook = Cookbook(recipe_path)
        # environment rl parameter
        self.actions = CraftWorldEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions) * neural_agent_num)
        self.del2direction = {(-1,0): int(CraftWorldEnv.Actions.left),
                          (1,0): int(CraftWorldEnv.Actions.right),
                          (0,-1): int(CraftWorldEnv.Actions.down),
                          (0,1): int(CraftWorldEnv.Actions.up),
                          (0,0): int(CraftWorldEnv.Actions.nothing)}
        self.direction2del = {x:y for y,x in self.del2direction.items()}
        
        # set up gui
        self._use_gui = use_gui
        if use_gui:
            self.gui = CraftGui(self, width, height,
                                is_headless=is_headless,
                                target_fps=target_fps)
        # TODO dsleeps: Fix the gui observation space
        if use_gui:
            self.observation_space = spaces.Tuple((
                spaces.Box(low=0, high=255,
                           shape=(80, 80, 3),
                           dtype=np.uint8),
                spaces.Box(low=0, high=time_limit,
                           shape=(self.cookbook.n_kinds+2+4, ),
                           dtype=np.float32),
                spaces.Box(low=-1, high=1.,
                           shape=(self.cookbook.n_kinds, 5),
                           dtype=np.float32))
            )
        else:
            self.n_features = \
                window_width * window_height * \
                (self.cookbook.n_kinds+1 + int(neural_agent_num) + len(self.cookbook.det_agent_names) - 1) + \
                self.cookbook.n_kinds + 4 + 1 # TODO dsleeps: For some reason this is off by 1
            self.observation_space = self.observation_space = spaces.Tuple((
                spaces.Box(low=0, high=time_limit,
                           shape=(self.n_features, ),
                           dtype=np.float32),
                spaces.Box(low=-1, high=1.,
                           shape=(self.cookbook.n_kinds, 5),
                           dtype=np.float32)
            ))
        self.prefix_reward_decay = prefix_reward_decay
        self.time_limit = time_limit
        self.update_failed_trans_only = update_failed_trans_only
        # convert the ltl formula to a Buchi automaton
        self._formulas = formulas
        self._alphabets = get_alphabets(recipe_path)
        #print(self.update_failed_trans_only)
        #exit()
        # TODO dsleeps: Implement this for multi-agent
        if self.update_failed_trans_only:
            ltl_tree = ltl2tree(self._formula, self._alphabets)
            self._anno_formula, _, self._anno_maps = ltl_tree_with_annotation_str(ltl_tree, idx=0)
            self.ba = Automaton(self._anno_formula, self._alphabets, add_flexible_state=False)
        else:
            self.bas = []
            self._last_states = []
            for formula in formulas:
                self.bas.append(Automaton(formula, self._alphabets, add_flexible_state=False))
        #self.ba.draw('tmp_images/ba.svg', show=False)
        #exit()
        self._seq = []
        # set the environment
        self._width = width
        self._height = height
        self._window_width = window_width
        self._window_height = window_height
        self.grid = copy.deepcopy(grid)
        # TODO dsleeps: Neural agent num does nothing, for now only support one agent
        self.neural_agent_num = int(neural_agent_num)
        self.det_agent_num = len(self.cookbook.det_agent_names)

        # The neural agents are first in the indices, then it's by the order of the list
        # The inventory object is shared across all agents, and each one knows its index
        self.inventories = np.zeros((self.neural_agent_num + self.det_agent_num, self.cookbook.n_kinds))

        self.workshop_outs = np.zeros(self.cookbook.n_kinds)#NOTE czw check
        self.approaching = [[]] * (self.neural_agent_num + self.det_agent_num) #NOTE czw check 
        
        # Create the agents
        self.load((grid, init_pos, init_dir), False)

        # start the first game
        self.reset()
        self.should_skip = False
        #print("init_done")

    def load(self, data, do_reset=True):
        self._init_grid = data[0]
        init_pos = data[1]
        init_dir = data[2]
        
        if (type(init_pos) != list):
            init_pos = [init_pos]
        if (type(init_dir) != list):
            init_dir = [init_dir]

        self.neural_agent = Agent.NeuralAgent(init_pos[0], init_dir[0], 
                                              self.inventories[0], 0, 'Neural_Agent_',
                                              self._formulas[0], 
                                              self.bas[0])
        
        self.det_agents = []
        initial_i = 1
        i = 1
        while (len(init_pos) != i):
            agent_type = self.cookbook.det_agent_types[i - initial_i]
            if (agent_type == 'Random'):
                self.det_agents.append(Agent.RandomAgent(init_pos[i], init_dir[i], self.inventories[i], i, 
                                                        self.cookbook.det_agent_names[i - initial_i] + "_", 
                                                        False))
            elif (agent_type == 'Pickup'):
                self.det_agents.append(Agent.PickupAgent(init_pos[i], init_dir[i], self.inventories[i], i, 
                                                        self.cookbook.det_agent_names[i - initial_i] + "_", 
                                                        self._width, self._height, False))
            i += 1
        
        # If positions haven't been specified for all of the non-neural agents
        # then randomize the remaining ones 
        while (len(self.det_agents) < self.det_agent_num):
            r_pos = (np.random.randint(self._width), np.random.randint(self._height))
            if (self.get_item(r_pos[0], r_pos[1]) == 0):
                agent_type = self.cookbook.det_agent_types[i - initial_i]
                if (agent_type == 'Random'):
                    # self.Actions - 1 ensures that I don't accidentally get a use action
                    self.det_agents.append( \
                         Agent.RandomAgent(r_pos, np.random.randint(len(self.Actions)-1), self.inventories[i], 
                                           i, self.cookbook.det_agent_names[i - initial_i] + "_", False))
                elif (agent_type == 'Pickup'):
                    self.det_agents.append( \
                         Agent.PickupAgent(r_pos, np.random.randint(len(self.Actions)-1), self.inventories[i], 
                                           i, self.cookbook.det_agent_names[i - initial_i] + "_", 
                                           self._width, self._height, False))
                i += 1
        
        # A list combining the two
        self.all_agents = [self.neural_agent] + self.det_agents
        
        if (do_reset):
            self.reset()

    def get_data(self):
        init_pos = [agent._init_pos for agent in self.all_agents]
        init_dir = [agent._init_dir for agent in self.all_agents]
        return self._init_grid, init_pos, init_dir

    def predicates(self, prev_ds, prev_inv, prev_workshop_outs, prev_approaching):
        pred = []
        # check to see the current colors of the scene
        for color_index in self.cookbook.color_indices:
            name = self.cookbook.index.get(color_index)
            if (np.sum(self.grid[:,:,color_index]) != 0):
                pred.append(name)
            else:
                pass
                # pred.append(name + '_off')
        
        # check to see if the light is on
        for switch_index in self.cookbook.switch_indices:
            name = self.cookbook.index.get(switch_index)
            light_state = np.sum(self.grid[:,:,switch_index]) # Taking advantage of the fact that there's only
                                                          # one light
            # A state of 1 is off
            if (light_state == 1):
                pred.append(name + '_on')
            else:
                pred.append(name + '_off')

        # check if the neighbors have env thing
        for i, agent in enumerate(self.all_agents):
            agent_str = agent.pred_str
            for nx, ny in nears(agent.pos, self._width, self._height):
                here = self.grid[nx, ny, :]
                if not self.get_item(nx, ny):
                    continue
                thing = here.argmax()
                if thing in self.cookbook.environment:
                    if 'recycle' in self.cookbook.index.get(thing):
                        continue  # temp for basic
                    if self.cookbook.index.get(thing) not in pred:
                        pred.append(agent_str + self.cookbook.index.get(thing))
            # check if any item in the inventory
            for thing, count in enumerate(agent.get_items()):
                name = self.cookbook.index.get(thing)
                if count > 0 and 'none' not in name:
                    pred.append(agent_str + name)
            # check if getting closer to an item
            ds = self.dist2items(agent)
            delta = ds - prev_ds
            delta_inv = agent.get_items() - prev_inv[i]
            delta_workshop_outs = self.workshop_outs - prev_workshop_outs
            for thing, d in enumerate(delta):
                name = self.cookbook.index.get(thing)
                if isinstance(agent.dir, np.ndarray):
                    agent.dir = agent.dir[0]
                if isinstance(agent.dir, torch.Tensor):
                    agent.dir = agent.dir[0].item()
                facing_del = self.direction2del[agent.dir]
                facing_pos = agent.pos[0] + facing_del[0], agent.pos[1] + facing_del[1]

                closer_to_thing = False
                #print(sum(delta_inv))
                if d < 0:
                    pred.append(agent_str + 'C_' + name)
                    closer_to_thing = True
                elif -0.1 < d < 0.1 and sum(delta_inv) != 0: #NOTE czw changed from > to !=
                    closer_pred = 'C_' + name
                    if closer_pred in prev_approaching:
                        pred.append(agent_str + 'C_' + name)
                        closer_to_thing = True
                elif facing_pos[0] in range(self._width) and facing_pos[1] in range(self._height):
                    facing_item = self.grid[facing_pos[0], facing_pos[1], :]
                    if -0.1 < d < 0.1 and facing_item.any() and facing_item.argmax() == thing:#NOTE czw: if facing the right item
                        pred.append(agent_str + 'C_' + name)
                        closer_to_thing = True
                if closer_to_thing:#NOTE:czw if item is in the bin, then we can be moving closer to it
                    if 'recycle' in name:
                        input_name = name.split('_')[1]
                        output_name = self.cookbook.input2output[input_name]
                        output_idx = self.cookbook.get_index(output_name)
                        if self.workshop_outs[output_idx] > 0:
                            pred.append(agent_str + 'C_' + input_name)
            self.approaching[i] = list(filter(lambda x: 'C_' in x, pred))
        return pred
    
    def filter_predicates(self, preds, agent):
        new_preds = []
        for pred in preds:
            if (agent.pred_str in pred):
                new_preds.append(pred[len(agent.pred_str):])
            else:
                new_preds.append(pred)
        return new_preds

    def step(self, action, no_eval=False):
        if (type(action) != list):
            action = [action]
        
        prev_inv = copy.deepcopy(self.inventories)
        prev_approaching = copy.deepcopy(self.approaching)
        prev_workshop_outs = copy.deepcopy(self.workshop_outs)
        for i, agent in enumerate(self.all_agents):
            prev_ds = self.dist2items(agent)
            x, y = agent.pos
            if (agent.is_neural):
                agent_action = action[i]
            else:
                agent_action = agent.take_action(self.grid, self.cookbook)
            n_dir = agent_action

            if agent_action == self.actions.left:
                dx, dy = (-1, 0)
            elif agent_action == self.actions.right:
                dx, dy = (1, 0)
            elif agent_action == self.actions.up:
                dx, dy = (0, 1)
            elif agent_action == self.actions.down:
                dx, dy = (0, -1)
            elif agent_action == self.actions.use:
                dx, dy = (0, 0)
                n_dir = agent.dir
            elif agent_action == self.actions.nothing:
                dx, dy = (0, 0)
                n_dir = agent.dir
            else:  # not supported move
                raise ValueError('Not supported action')

            # move
            agent.dir = n_dir
            x = x + dx
            y = y + dy

            if x in range(0, self._width) and y in range(0, self._height) and \
               not self.get_item(x, y) and not (x, y) in [a.pos for a in self.all_agents]:
                agent.pos = (x, y)
            # take `use` action
            if agent_action == self.actions.use:
                success = False
                for nx, ny in neighbors(agent.pos, self._width, self._height, agent.dir):
                    if not self.get_item(nx, ny):
                        continue
                    thing = self.get_item(nx, ny)
                    #print("thing", self.cookbook.index.get(thing))
                    #print(self.cookbook.workshop_indices)
                    if not(thing in self.cookbook.grabbable_indices or \
                            thing in self.cookbook.workshop_indices or \
                            thing in self.cookbook.switch_indices or \
                            thing == self.cookbook.water_index or \
                            thing == self.cookbook.stone_index):
                        continue
                    if thing in self.cookbook.grabbable_indices:
                        agent.add_items(thing, self.grid)
                        self.grid[nx, ny, thing] = 0
                        success = True
                    elif thing in self.cookbook.workshop_indices:
                        workshop = self.cookbook.index.get(thing)
                        for output, inputs in self.cookbook.recipes.items():
                            if inputs["_at"] != workshop:
                                continue
                            for key in inputs.keys():
                                if key == '_at':
                                    continue
                                if agent.get_items()[key] == 0 and self.workshop_outs[output] == 0:
                                    continue
                                if agent.get_items()[key] >= inputs[key]:
                                    agent.get_items()[key] -= inputs[key]
                                    self.workshop_outs[output] += 1
                                elif self.workshop_outs[output] > 0:
                                    agent.get_items()[key] += 1
                                    self.workshop_outs[output] -= 1
                            success = True
                    elif thing in self.cookbook.switch_indices:
                        # Flip the light switch on and off
                        # 1 is off and 2 is on
                        # TODO: There are asserts all over the place asserting that things
                        #       sum to one. Maybe get rid of those?
                        self.grid[nx, ny, thing] = 2 if self.grid[nx, ny, thing] == 1 else 1
                    elif thing == self.cookbook.water_index:
                        if agent.get_items()[self.cookbook.index["bridge"]] > 0:
                            self.grid[nx, ny, self.water_index] = 0
                            agent.get_items()[self.cookbook.index["bridge"]] -= 1
                    elif thing == self.cookbook.stone_index:
                        if agent.get_items()[self.cookbook.index["axe"]] > 0:
                            self.grid[nx, ny, self.stone_index] = 0
                    break
        
        # Randomly change the colors if any of them exist
        for color_index in self.cookbook.color_indices:
            # TODO Define this somewhere else
            change_prob = 0.1
            if (np.random.rand() < change_prob):
                if (np.sum(self.grid[:,:,color_index]) != 0):
                    self.grid[:,:,color_index] = 0
                else:
                    self.grid[:,:,color_index] = 1

        if no_eval:
            return

        # get predicates
        trans = self.predicates(prev_ds, prev_inv, prev_workshop_outs, prev_approaching)
        self.neural_agent.seq.append(self.filter_predicates(trans, agent))
        
        done = len(self.neural_agent.seq) >= self.time_limit
        
        # check if it is prefix or accepting state
        reward = 0
        is_prefix, dist_to_accept, last_states, failed_trans = \
                self.neural_agent.ba.is_prefix([self.neural_agent.seq[-1]], self.neural_agent.last_states)
        is_accept = is_prefix and dist_to_accept < 0.1
        if is_accept:  # not done even if it is in accept state
            reward = 1
            self.neural_agent.last_states = set([s for s in last_states if self.neural_agent.ba.is_accept(s)])
        elif is_prefix:
            if len(last_states.intersection(self.neural_agent.last_states)) > 0:
                self.neural_agent.state_visit_count += 1
            else:
                self.neural_agent.state_visit_count = 1
            self.neural_agent.last_states = last_states
            if self.neural_agent.state_visit_count == 1:
                reward = 0.1
            else:
                reward = 0.1 * (self.prefix_reward_decay ** (self.neural_agent.state_visit_count - 1))
        else:
            reward = -1
            done = True
        if done and reward < 0.2:  # penalize if reaching time limit but not accept
            reward = -1
        
        if self._use_gui:
            self.gui.draw()
        info = {}
        if self.update_failed_trans_only:
            components = set()
            for trans in failed_trans:
                for symbol in trans.split(' '):
                    symbol = symbol.replace('(', '').replace(')', '').replace('!', '')
                    if symbol in LTL_OPS: continue
                    if 'a_' in symbol:
                        components = components.union(self._anno_maps[symbol])
            info = {'failed_components': components}
        
        # print(rewards)
        return self.feature(), reward, done, info
    
    # Have to update this function for every element that isn't an "item"
    def get_item(self, x, y):
        item_indices = np.where(self.grid[x, y, :] == 1)
        for i in item_indices[0]:
            if not i in self.cookbook.color_indices:
                return i
        return 0

    def dist2items(self, agent):
        # no need to -2 if no boundary
        total_ds = float(self._width + self._height)
        min_ds = np.ones(self.cookbook.n_kinds) * total_ds
        for y in reversed(range(self._height)):
            for x in range(self._width):
                if not (self.get_item(x, y) or (x, y) == agent.pos):
                    continue
                else:
                    thing = self.get_item(x, y)
                    d = abs(x - agent.pos[0]) + abs(y - agent.pos[1])
                    #print(thing)
                    #print(self.cookbook.index.get(thing))
                    if min_ds[thing] > d:
                        min_ds[thing] = d
        # distance is zero if own the item
        for i, count in enumerate(agent.get_items()):
            if count > 0:
                min_ds[i] = 0
        return min_ds

    def closer_feature(self, agent):
        ds = []
        # remember the current config
        current_ds = self.dist2items(agent)
        current_grid = copy.deepcopy(self.grid)
        current_inventory = copy.deepcopy(agent.get_items())
        current_workshop_outs = copy.deepcopy(self.workshop_outs)
        current_approaching = copy.deepcopy(self.approaching[agent.inv_index])
        #print("current_aproaching", list(current_approaching))
        current_pos = copy.deepcopy(agent.pos)
        current_dir = copy.deepcopy(agent.dir)
        for action in range(5):
            # take step
            actions = [5] * self.neural_agent_num
            if (agent.is_neural):
                actions[agent.inv_index] = action
            self.step(actions, no_eval=True)
            ds.append(current_ds-self.dist2items(agent))

            # restore the current grid
            self.grid = copy.deepcopy(current_grid)
            self.inventories[agent.inv_index] = copy.deepcopy(current_inventory)
            self.workshop_outs = copy.deepcopy(current_workshop_outs)
            self.approaching[agent.inv_index] = copy.deepcopy(current_approaching)
            agent.pos = copy.deepcopy(current_pos)
            agent.dir = copy.deepcopy(current_dir)
        ds = np.asarray(ds)
        return np.transpose(ds, (1, 0))

    def feature(self):
        # TODO dsleeps: Fix the gui code in the feature function
        if self._use_gui:
            # position features
            pos_feats = np.asarray(agent.pos).astype(np.float32)
            pos_feats[0] /= self._width
            pos_feats[1] /= self._height
            # direction features
            dir_features = np.zeros(5)
            dir_features[agent.dir] = 1
            
            hw = int(self._window_width / 2)
            hh = int(self._window_height / 2)
            img_str = pygame.image.tostring(self.gui._screen, 'RGB')
            img = Image.frombytes('RGB', self.gui._screen.get_size(), img_str)
            cell_width = int(self.gui._cell_width); cell_height = int(self.gui._cell_height)
            px_x = x * cell_width; px_y = (self._height - y - 1) * cell_height
            px_hw = int(hw * cell_width); px_hh = int(hh * cell_height)
            img_padded = pad(img, pad_width=((px_hw, px_hw),
                                             (px_hh, px_hh),
                                             (0,0)),
                             mode='constant')
            new_x = px_hw + px_x; new_y = px_hh + px_y
            out_img = img_padded[new_y-px_hh:new_y+px_hh+cell_height, \
                                 new_x-px_hw:new_x+px_hw+cell_width]
            out_img = out_img.reshape((px_hh*2+cell_height, px_hw*2+cell_width, 3))
            out_img = resize(out_img, [80, 80, 3],
                             preserve_range=True, anti_aliasing=True)
            out_values = np.concatenate((agent.get_items(), pos_feats, dir_features))
            features = {0: out_img.astype(np.uint8), 1: out_values, 2: self.closer_feature(agent), 3: img}
        else:
            hw = int(self._window_width / 2)
            hh = int(self._window_height / 2)
            bhw = int((self._window_width * self._window_width) / 2)
            bhh = int((self._window_height * self._window_height) / 2)
            
            features = {}
            new_grid = np.concatenate((self.grid, np.zeros((self._width, self._height, len(self.all_agents)-1)))
                                      , axis=2)
            
            # TODO dsleeps: Currently doesn't get the direction of the agents
            i = 0
            x, y = self.neural_agent.pos
            dir_features = np.zeros(5)
            dir_features[self.neural_agent.dir] = 1
            
            # Add each other neural agent to the grid
            grid_copy = copy.deepcopy(new_grid)
            passed = 0
            for j, n_agent in enumerate(self.all_agents):
                # This skips the current agent
                if j == i:
                    passed = 1
                    continue
                n_x, n_y = n_agent.pos
                grid_copy[n_x, n_y, j-passed] = 1

            grid_feats = pad_slice(grid_copy, (x-hw, x+hw+1), 
                    (y-hh, y+hh+1))
            out_grid = np.concatenate((grid_feats.ravel(),
                                       self.neural_agent.get_items(),
                                       dir_features))
            features[2*i] = out_grid
            features[2*i + 1] = self.closer_feature(self.neural_agent)
        return features

    def reset(self):
        self.workshop_outs = np.zeros(self.cookbook.n_kinds)
        self.approaching = [[]] * len(self.all_agents)
        self.grid = copy.deepcopy(self._init_grid)
        
        for agent in self.all_agents:
            agent.reset()

        if self._use_gui:
            self.gui.draw()
        # TODO: What is self.feature() used for?
        return self.feature()

    def visualize(self):
        s = ''
        for y in reversed(range(self._height)):
            for x in range(self._width):
                if not (self.get_item(x, y) or (x, y) == self.pos):
                    ch = ' '
                else:
                    thing = self.get_item(x, y)
                    if (x, y) == self.pos:
                        if self.dir == self.actions.left:
                            ch = "<"
                        elif self.dir == self.actions.right:
                            ch = ">"
                        elif self.dir == self.actions.up:
                            ch = "^"
                        elif self.dir == self.actions.down:
                            ch = "v"
                    elif thing == self.cookbook.get_index("boundary"):
                        ch = 'X'
                    else:
                        ch = chr(97+thing)
                s += ch
            s += '\n'
        print(s)

    def print_inventory(self):
        print('Current inventory items:')
        for agent in self.all_agents:
            print('Agent #' + str(agent.inv_index))
            for item, count in enumerate(agent.get_items()):
                if count > 0:
                    print('{}: {}'.format(self.cookbook.index.get(item), count))
        print('Current workshop outs:')
        for item, count in enumerate(self.workshop_outs):
            if count > 0:
                print('{}: {}'.format(self.cookbook.index.get(item), count))
        print('----------')


def get_alphabets(recipe_path):
    cookbook = Cookbook(recipe_path)
    alphabets = []
    for item in cookbook.index:
        if 'none' in item or 'recycle' in item:
            continue
        alphabets.append(item)
        alphabets.append('C_'+item)  # the predicate for closer to item
    return alphabets


def check_excluding_formula(formula, alphabets, recipe_path):
    ba = Automaton(formula, alphabets)
    cookbook = Cookbook(recipe_path)
    
    def check_multi_env(label):
        if '|' not in label and '&' not in label:
            return True
        elif '|' in label:
            for l in label.split(' | '):
                if not check_multi_env(l):
                    return False
        elif '&' in label:
            has_env = False
            for l in label.lstrip('(').rstrip(')').split(' & '):
                if '!' in l:
                    continue
                if cookbook.get_index(l) in cookbook.environment:
                    if not has_env:
                        has_env = True
                    else:
                        return False
        return True

    # only one env in each state
    for s in range(ba.n_states):
        dsts, labels = ba.get_transitions(str(s))
        for label in labels:
            if not check_multi_env(label):
                return False
    return True

# TODO dsleeps: Might have to remove the grid.any and replace it with get_item
def random_free(grid, rand, width, height):
    pos = None
    while pos is None:
        (x, y) = (rand.randint(width), rand.randint(height))
        if grid[x, y, :].any():
            continue
        # check if nearby is occupied
        ns = neighbors((x,y), width, height)
        occupied = 0
        for n in ns:
            if grid[n[0], n[1], :].any():
                occupied += 1
        if occupied > 2: #== len(ns)/2:
            continue
        pos = (x, y)
    return pos

# TODO dsleeps: Might have to remove the grid.any and replace it with get_item
def sample_craft_env_each(args, width=7, height=7, env_data=None, env=None):
    #print("sample")
    #print(args.formula)
    cookbook = Cookbook(args.recipe_path)
    rand = np.random.RandomState()
    # generate grid
    grid = np.zeros((width, height, cookbook.n_kinds+1))
    '''
    i_bd = cookbook.index["boundary"]
    if i_bd is not None:
        grid[0, :, i_bd] = 1
        grid[width-1:, :, i_bd] = 1
        grid[:, 0, i_bd] = 1
        grid[:, height-1:, i_bd] = 1
    '''
    if env_data is None:
        # add treasure
        # 1) gold and water (needs to make bridge)
        # 2) gem and stone (needs to make cave)
        if 'basic' not in args.recipe_path:
            wall_names = ['water', 'stone']
            for i, treasure in enumerate(['gold', 'gem']):
                (gx, gy) = random_free(grid, rand, width, height)
                treasure_index = cookbook.index[treasure]
                wall_index = cookbook.index[wall_names[i]]
                grid[gx, gy, treasure_index] = 1
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if not grid[gx+i, gy+j, :].any() and \
                                gx+i >= 0 and gx+i < width-1 and \
                                gy+j >= 0 and gy+j < height-1:
                            grid[gx+i, gy+j, wall_index] = 1
        # ingredients
        #czw sparse
        primitives = []
        #print("primitives")
        #print(cookbook.primitives)
        primitives = [p for p in cookbook.primitives if cookbook.index.get(p) in args.formula or random.random() < 0.3]#TODO alpha
        #primitives = cookbook.primitives
        for primitive in primitives:
            if 'basic' not in args.recipe_path and \
                    (primitive == cookbook.index["gold"] or \
                    primitive == cookbook.index["gem"]):
                continue
            for i in range(1):
                (x, y) = random_free(grid, rand, width, height)
                grid[x, y, primitive] = 1
        # generate crafting stations
        if 'basic' not in args.recipe_path:
            for i_ws in range(3):
                ws_x, ws_y = random_free(grid, rand, width, height)
                grid[ws_x, ws_y, cookbook.index["workshop%d" % i_ws]] = 1
        else:
            environments = []
            for p in cookbook.environment:
                if cookbook.index.get(p) in args.formula or random.random() < 0.3:#TODO alpha
                    environments.append(p)
                elif p in cookbook.workshop_indices:
                    bin_input = cookbook.index.get(p).split("_")[1]
                    input_idx = cookbook.index[bin_input]
                    if input_idx in primitives:
                        environments.append(p)
            #print(args.formula)
            #environments.append(cookbook.index["boundary"])#czw
            #print([cookbook.index.get(e) for e in environments])
            #exit()
            #print([cookbook.index.get(e) for e in environments])
            for s_env in environments:
                if s_env == cookbook.get_index('boundary'):
                    continue
                (x, y) = random_free(grid, rand, width, height)
                grid[x, y, s_env] = 1
        # generate init pos
        init_pos = [random_free(grid, rand, width, height) for i in range(args.neural_agent_num)]
        init_dir = [rand.randint(4) for i in range(args.neural_agent_num)]
        env_data = (grid, init_pos, init_dir)
    else:
        grid, init_pos, init_dir = env_data
    # return the env
    #print(env)
    if env is not None:
        env.load(env_data)
        return env
    else:
        return CraftWorldEnv([args.formula] * args.neural_agent_num, args.recipe_path,
                             init_pos, init_dir, grid, neural_agent_num=args.neural_agent_num,
                             width=width, height=height,
                             use_gui=args.use_gui,
                             is_headless=args.is_headless,
                             prefix_reward_decay=args.prefix_reward_decay,
                             time_limit=args.num_steps,
                             target_fps=args.target_fps,
                             update_failed_trans_only=args.update_failed_trans_only)

def bfs(env, goal, agent):
    #print("bfs")
    #print("goal")
    #print(env.cookbook.index.get(goal))
    visited = set()
    queue = [(agent.pos, [])]
    delts = env.del2direction.keys()
    while len(queue) > 0:
        curr, curr_seq = queue.pop(0)
        visited.add(curr)
        x, y = curr
        for d in delts:
            x_n, y_n = x+d[0], y+d[1]
            if not (x_n in range(env._width) and y_n in range(env._height)):
                continue
            if not env.get_item(x_n, y_n):
                if (x_n, y_n) not in visited:
                    new_seq = curr_seq.copy()
                    new_seq.append(env.del2direction[d])
                    queue.append(((x_n, y_n), new_seq))
                continue
            thing = env.get_item(x_n, y_n)
            #print(env.cookbook.index.get(goal))
            #print()
            if thing == goal:
                curr_seq.append(env.del2direction[d])
                curr_seq.append(int(CraftWorldEnv.Actions.use))
                #print(curr_seq)
                return True, curr_seq, env.del2direction[d]
    return False, [], None

def use_item(env, item, agent, drop=True):
    #print(env.grid)
    item_name = env.cookbook.index.get(item)
    if drop:
        item_name = "recycle_" + item_name
    if 'none' in item_name:
        item_name = "recycle_" + env.cookbook.output2input[item_name]
    item_idx = env.cookbook.index[item_name]
    success, seq, last_action = bfs(env, item_idx, agent)
    if success:
        return seq
    return []    

def get_items(env):
    things = []
    for i in range(env._height):
        for j in range(env._width):
            if env.get_item(i, j):
                thing = env.get_item(i, j)
                things.append(thing)
    return things

# TODO dsleeps: Not sure if this works right for multiple agents. Maybe this should just be for 1?
def gen_actions(env, max_n_seq, agent, goal_only=True, n_tracks=1, for_dataset=False):
    '''
        If there are at least n_tracks successful tracks return:
            True, [(actions_1, first_accept_1), (actions_2, first_accept_2), ...]
            where:
                actions_i is a sequence of env.time_limit actions
                first_accept_i is the first time step at which the NFA persistently accept
            The entire list is sorted by increasing first_accept_i
        
        NOTE:
            For the purposes of training the agent, INSTEAD return:
                True, actions
                where:
                    actions is a sequence of env.time_limit actions
    '''
    #sampled_actions = np.random.choice(5, (max_n_seq, env.time_limit))
    seqs = []
    for i in range(max_n_seq):
        env.reset()
        found_len = 100
        step_prob, stay_prob = 0.33, 0.33#NOTE tune
        alpha = 0.8
        step_j = 0
        actions = []
        last_waypoint_action = None
        violated = False
        #print("seq****", i)
        while len(actions) < env.time_limit:
            #print("step", step_j)
            subtask = []
            diff = env.time_limit - len(actions)
            random_length = random.choice(range(diff)) + 1
            dice_roll = random.random()
            if dice_roll < step_prob:
                #print("step")
                subtask = np.random.choice(5, (random_length)).tolist()
                last_waypoint_action = subtask[-1]
            elif step_prob < dice_roll < stay_prob + step_prob and last_waypoint_action:
                #print("stay")
                subtask = [last_waypoint_action]*random_length
            else:
                #print("sub")
                # Get all of the items in the environment
                items = [j for j in range(len(agent.get_items())) if agent.get_items()[j] > 0]
                items += [j for j in range(len(env.workshop_outs)) if env.workshop_outs[j] > 0]
                items += get_items(env)
                items = list(filter(lambda x: x not in env.cookbook.workshop_indices, items))
                
                # TODO dsleeps: Idk if this is the right thing to do
                if (len(items) != 0):
                    item = random.choice(items)
                    drop = agent.get_items()[item] > 0
                    #print([env.cookbook.index.get(i) for i in items])
                    subtask = use_item(env, item, agent, drop=drop)
                    if len(subtask) > 1: 
                        last_waypoint_action = subtask[-2]
                
            subtask = subtask[:diff]
            actions += subtask

            #print(subtask)
            for j in range(len(subtask)):
                action = [5] * env.neural_agent_num
                action[agent.inv_index] = subtask[j]
                obs, rewards, done, _ = env.step(action)
                reward = rewards
                # reward = rewards[agent.inv_index] # TODO dsleeps: Make this work with multiple agents
                if reward < 0.9:
                    found_len = step_j
                if done:
                    if (goal_only and reward < 0.9) or (not goal_only and reward > 0):
                        if (for_dataset):
                            return True, (actions, found_len,)
                        else:
                            return True, actions
                        #seqs.append((actions, found_len))#TODO change back
                step_j += 1
                if reward < 0:
                    violated = True
                    break
            if violated:
                break
    return False, []

# TODO dsleeps: Not sure if this works for multiple agents
def sample_craft_env(args, width=7, height=7, n_steps=1, k_path=5,
                     n_retries=5, env_data=None, train=False, check_action=True,
                     max_n_seq=50, goal_only=True, n_tracks=1, for_dataset=False, 
                     neural_agent_num=1, det_agent_num=0):
    no_env = True; env = None; count = 0; actions = None
    # exclude the env that is true at beginning
    while no_env:
        env = sample_craft_env_each(args, width, height, env_data, env)
        grid = env.grid
        actions = []
        for i in range(k_path):  # try k_path random paths
            #<czw>
            env.reset()
            if i < 5:#test to make sure that no 
                action = [i] * args.neural_agent_num
                obs, rewards, done, _ = env.step(action)
                rewards = [rewards]
                no_env = sum(rewards) > 0.5 * len(rewards) #or (j == 0 and done) #TODO dsleeps: This is bad
                if no_env:
                    break
            #</czw>
            env.reset()
            for j in range(n_steps):
                action = [np.random.choice(5) for k in range(args.neural_agent_num)]
                obs, rewards, done, _ = env.step(action)
                rewards = [rewards]
                # not sample an env if too easy or fail at first step
                no_env = sum(rewards) > 0.5 * len(rewards) #or (j == 0 and done) #TODO dsleeps: This is bad
                if no_env:
                    break
            if no_env:
                break
        if not no_env and check_action:  # filter out the env that doesn't have a solution
            env.reset()
            success, agent_actions = gen_actions(env, max_n_seq=max_n_seq, agent=env.neural_agent, 
                                           goal_only=goal_only, n_tracks=1, for_dataset=for_dataset)
            actions.append(agent_actions)
            if not success:
                no_env = True
        count += 1
        if count > n_retries:  # retry too many times
            break
    if no_env:
        env.should_skip = True
    env.reset()
    if not train and no_env:
        return None, [[]] * neural_agent_num
    else:
        return env, actions


def one_hot(grid, n_features):
    width, height = grid.shape
    one_hot = np.zeros((width*height, n_features))
    flat_grid = grid.flatten()
    one_hot[np.arange(width*height), flat_grid] = 1
    for row in one_hot:
        assert row.sum() == 1
    one_hot[:,0] = 0 #0 encoded as 0
    one_hot = one_hot.reshape((width, height, cookbook.n_kinds+1))
    return one_hot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Craft world')
    args = parser.parse_args()
    args.recipe_path = ROOT_PATH + 'craft_recipes_basic.yaml'
    # args.recipe_path = ROOT_PATH + 'craft_recipes_basic_color.yaml'
    args.num_steps = 20
    args.target_fps = 60
    args.use_gui = False
    args.is_headless = True
    # args.formula = "( G ( tree ) )"#NOTE dummy formula
    # args.formula = "(( tree ) U ! ( blue ) )"
    args.formula = "( F ( tree ) )"
    args.prefix_reward_decay = 0.8
    args.update_failed_trans_only = False
    args.return_screen = False
    args.neural_agent_num = 1
    args.det_agent_num = 2
    env, actions = sample_craft_env(args, n_steps=3, n_retries=10, train=True, neural_agent_num=2, det_agent_num=5)
    if env is None: exit()
    env.neural_agent.ba.draw('tmp_images/agent_' + str(0) + '_ba.svg', show=False)
    '''
    while True:
        env.gui.draw(move_first=True)
        feature = env.feature()
        img = Image.fromarray(feature[0])
        img.save('tmp_images/feature.png')
    '''
