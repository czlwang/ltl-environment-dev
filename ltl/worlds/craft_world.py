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

ROOT_PATH = "/Users/dsleeps/Documents/ltl-environment-dev/ltl/worlds/" #TODO adjust this

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
        self.recipes = {}
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
        for direction in ['left', 'right', 'up', 'down']:
            #combos = ['apple', 'apple_orange', 'apple_pear', 'all', 'empty', 'orange', 'orange_pear', 'pear']
            combos = ['gem', 'gem_gold', 'gem_iron', 'all', 'empty', 'gold', 'gold_iron', 'iron']
            for combo in combos:
                filename = img_path +  direction + "_" + combo + '.png'
                d = self._sprites.get(direction, {})
                d[combo] = pygame.image.load(filename)
                self._sprites[direction] = d
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
                    obs, reward, done, _ = self._env.step(self._env.actions.up)
                elif event.key == pygame.K_DOWN:
                    print(self._env.actions.down)
                    obs, reward, done, _ = self._env.step(self._env.actions.down)
                elif event.key == pygame.K_LEFT:
                    print(self._env.actions.left)
                    obs, reward, done, _ = self._env.step(self._env.actions.left)
                elif event.key == pygame.K_RIGHT:
                    print(self._env.actions.right)
                    obs, reward, done, _ = self._env.step(self._env.actions.right)
                elif event.key == pygame.K_SPACE:
                    #print("about to step*******")
                    print(self._env.actions.use)
                    obs, reward, done, _ = self._env.step(self._env.actions.use)
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
                if self._env.grid[x, y, :].any() or (x, y) == self._env.pos:
                    thing = self._env.grid[x, y, :].argmax()
                    if (x, y) == self._env.pos:
                        #objs = self.
                        obj_str = ''
                        #for obj in ["apple", "orange", "pear"]:
                        for obj in ["gem", "gold", "iron"]:
                            obj_idx = self._env.cookbook.index[obj]
                            if self._env.inventory[obj_idx] > 0:
                                obj_str += obj+"_"
                        obj_str = obj_str[:-1]
                        if len(obj_str) == 0:
                            obj_str = 'empty'
                        #if obj_str == 'apple_orange_pear':
                        if obj_str == 'gem_gold_iron':
                            obj_str = 'all'
                        #dir_sprite_dict = self._sprite[self._env.dir]:
                        #    obj_dir_sprite = dir_sprite_dict[objs]
                        if self._env.dir == self._env.actions.left:
                            picture = pygame.transform.scale(self._sprites['left'][obj_str], cell_size)
                        elif self._env.dir == self._env.actions.right:
                            picture = pygame.transform.scale(self._sprites['right'][obj_str], cell_size)
                        elif self._env.dir == self._env.actions.up:
                            picture = pygame.transform.scale(self._sprites['up'][obj_str], cell_size)
                        elif self._env.dir == self._env.actions.down:
                            picture = pygame.transform.scale(self._sprites['down'][obj_str], cell_size)
                    elif thing == self._env.cookbook.get_index("boundary"):
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
        use   = 4 # use

    def __init__(self, formula, recipe_path,
                 init_pos, init_dir, grid,
                 width=7, height=7, window_width=5, window_height=5,
                 prefix_reward_decay=1., time_limit=10,
                 use_gui=True, target_fps=None, is_headless=False,
                 update_failed_trans_only=False):
        self.first_accept = 100
        self.cookbook = Cookbook(recipe_path)
        # environment rl parameter
        self.actions = CraftWorldEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.del2direction = {(-1,0): int(CraftWorldEnv.Actions.left),
                          (1,0): int(CraftWorldEnv.Actions.right),
                          (0,-1): int(CraftWorldEnv.Actions.down),
                          (0,1): int(CraftWorldEnv.Actions.up)}
        self.direction2del = {x:y for y,x in self.del2direction.items()}

        # set up gui
        self._use_gui = use_gui
        if use_gui:
            self.gui = CraftGui(self, width, height,
                                is_headless=is_headless,
                                target_fps=target_fps)
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
                window_width * window_height * (self.cookbook.n_kinds+1) + \
                self.cookbook.n_kinds + 4
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
        self._formula = formula
        self._alphabets = get_alphabets(recipe_path)
        #print(self.update_failed_trans_only)
        #exit()
        if self.update_failed_trans_only:
            ltl_tree = ltl2tree(self._formula, self._alphabets)
            self._anno_formula, _, self._anno_maps = ltl_tree_with_annotation_str(ltl_tree, idx=0)
            self.ba = Automaton(self._anno_formula, self._alphabets, add_flexible_state=False)
        else:
            self.ba = Automaton(formula, self._alphabets, add_flexible_state=False)

        #self.ba.draw('tmp_images/ba.svg', show=False)
        #exit()
        self._seq = []
        self._last_states = set(self.ba.get_initial_state())
        self._state_visit_count = 0
        # set the environment
        self._width = width
        self._height = height
        self._window_width = window_width
        self._window_height = window_height
        self._init_pos = init_pos
        self._init_dir = init_dir
        self._init_grid = grid
        self.pos = copy.deepcopy(init_pos)
        self.dir = copy.deepcopy(init_dir)
        self.grid = copy.deepcopy(grid)
        self.inventory = np.zeros(self.cookbook.n_kinds)
        self.workshop_outs = np.zeros(self.cookbook.n_kinds)#NOTE czw check
        self.approaching = []#NOTE czw check
        # start the first game
        self.reset()
        self.should_skip = False
        #print("init_done")

    def load(self, data):
        self._init_grid = data[0]
        self._init_pos = data[1]
        self._init_dir = data[2]
        self.reset()

    def get_data(self):
        return self._init_grid, self._init_pos, self._init_dir

    def predicates(self, prev_ds, prev_inv, prev_workshop_outs, prev_approaching):
        pred = []
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
        for nx, ny in nears(self.pos, self._width, self._height):
            here = self.grid[nx, ny, :]
            if not self.grid[nx, ny, :].any():
                continue
            #print(here)
            #print(here.sum())
            # Count the number of nonzero indices
            # assert here.sum() == 1
            assert np.count_nonzero(here) == 1
            thing = here.argmax()
            if thing in self.cookbook.environment:
                if 'recycle' in self.cookbook.index.get(thing):
                    continue  # temp for basic
                if self.cookbook.index.get(thing) not in pred:
                    pred.append(self.cookbook.index.get(thing))
        # check if any item in the inventory
        for thing, count in enumerate(self.inventory):
            name = self.cookbook.index.get(thing)
            if count > 0 and 'none' not in name:
                pred.append(name)
        # check if getting closer to an item
        ds = self.dist2items()
        delta = ds - prev_ds
        delta_inv = self.inventory - prev_inv
        delta_workshop_outs = self.workshop_outs - prev_workshop_outs
        for thing, d in enumerate(delta):
            name = self.cookbook.index.get(thing)
            facing_del = self.direction2del[self.dir]
            facing_pos = self.pos[0] + facing_del[0], self.pos[1] + facing_del[1]

            closer_to_thing = False
            #print(sum(delta_inv))
            if d < 0:
                pred.append('C_' + name)
                closer_to_thing = True
            elif -0.1 < d < 0.1 and sum(delta_inv) != 0: #NOTE czw changed from > to !=
                closer_pred = 'C_' + name
                if closer_pred in prev_approaching:
                    pred.append('C_' + name)
                    closer_to_thing = True
            elif facing_pos[0] in range(self._width) and facing_pos[1] in range(self._height):
                facing_item = self.grid[facing_pos[0], facing_pos[1], :]
                if -0.1 < d < 0.1 and facing_item.any() and facing_item.argmax() == thing:#NOTE czw: if facing the right item
                    pred.append('C_' + name)
                    closer_to_thing = True
            if closer_to_thing:#NOTE:czw if item is in the bin, then we can be moving closer to it
                if 'recycle' in name:
                    input_name = name.split('_')[1]
                    output_name = self.cookbook.input2output[input_name]
                    output_idx = self.cookbook.get_index(output_name)
                    if self.workshop_outs[output_idx] > 0:
                        pred.append('C_' + input_name)
        self.approaching = list(filter(lambda x: 'C_' in x, pred))
        return pred

    def step(self, action, no_eval=False):
        prev_ds = self.dist2items()
        prev_inv = copy.deepcopy(self.inventory)
        prev_approaching = copy.deepcopy(self.approaching)
        prev_workshop_outs = copy.deepcopy(self.workshop_outs)
        x, y = self.pos
        n_dir = action
        if action == self.actions.left:
            dx, dy = (-1, 0)
        elif action == self.actions.right:
            dx, dy = (1, 0)
        elif action == self.actions.up:
            dx, dy = (0, 1)
        elif action == self.actions.down:
            dx, dy = (0, -1)
        elif action == self.actions.use:
            dx, dy = (0, 0)
            n_dir = self.dir
        else:  # not supported move
            raise ValueError('Not supported action')
        # move
        self.dir = n_dir
        x = self.pos[0] + dx
        y = self.pos[1] + dy
        if x in range(0, self._width) and y in range(0, self._height) and not self.grid[x, y, :].any():
            self.pos = (x, y)
        # take `use` action
        if action == self.actions.use:
            success = False
            for nx, ny in neighbors(self.pos, self._width, self._height, self.dir):
                here = self.grid[nx, ny, :]
                if not self.grid[nx, ny, :].any():
                    continue
                assert np.count_nonzero(here) == 1
                thing = here.argmax()
                #print("thing", self.cookbook.index.get(thing))
                #print(self.cookbook.workshop_indices)
                if not(thing in self.cookbook.grabbable_indices or \
                        thing in self.cookbook.workshop_indices or \
                        thing in self.cookbook.switch_indices or \
                        thing == self.cookbook.water_index or \
                        thing == self.cookbook.stone_index):
                    continue
                if thing in self.cookbook.grabbable_indices:
                    self.inventory[thing] += 1
                    self.grid[nx, ny, thing] = 0
                    success = True
                elif thing in self.cookbook.workshop_indices:
                    workshop = self.cookbook.index.get(thing)
                    for output, inputs in self.cookbook.recipes.items():
                        if inputs["_at"] != workshop:
                            continue
                        for i in inputs.keys():
                            if i == '_at':
                                continue
                            if self.inventory[i] == 0 and self.workshop_outs[output] == 0:
                                continue
                            if self.inventory[i] >= inputs[i]:
                                self.inventory[i] -= inputs[i]
                                self.workshop_outs[output] += 1
                            elif self.workshop_outs[output] > 0:
                                self.inventory[i] += 1
                                self.workshop_outs[output] -= 1
                        success = True
                elif thing in self.cookbook.switch_indices:
                    # Flip the light switch on and off
                    # 1 is off and 2 is on
                    # TODO: There are asserts all over the place asserting that things
                    #       sum to one. Maybe get rid of those?
                    self.grid[nx, ny, thing] = 2 if self.grid[nx, ny, thing] == 1 else 1
                elif thing == self.cookbook.water_index:
                    if self.inventory[self.cookbook.index["bridge"]] > 0:
                        self.grid[nx, ny, self.water_index] = 0
                        self.inventory[self.cookbook.index["bridge"]] -= 1
                elif thing == self.cookbook.stone_index:
                    if self.inventory[self.cookbook.index["axe"]] > 0:
                        self.grid[nx, ny, self.stone_index] = 0
                break
        if no_eval:
            return

        # get predicates
        trans = self.predicates(prev_ds, prev_inv, prev_workshop_outs, prev_approaching)
        self._seq.append(trans)
        done = len(self._seq) >= self.time_limit
        #print(len(self._seq), self.time_limit)
        # check if it is prefix or accepting state
        #print("trans")
        #print(trans)
        #exit()
        is_prefix, dist_to_accept, last_states, failed_trans = \
                self.ba.is_prefix([self._seq[-1]], self._last_states)
        #print("last states", self._last_states)
        #print("is_prefix", is_prefix)
        is_accept = is_prefix and dist_to_accept < 0.1
        if is_accept:  # not done even if it is in accept state
            reward = 1
            self._last_states = set([s for s in last_states if self.ba.is_accept(s)])
        elif is_prefix:
            if len(last_states.intersection(self._last_states)) > 0:
                self._state_visit_count += 1
            else:
                self._state_visit_count = 1
            self._last_states = last_states
            if self._state_visit_count == 1:
                reward = 0.1
            else:
                reward = 0.1 * (self.prefix_reward_decay ** (self._state_visit_count - 1))
        else:
            reward = -1
            done = True
        if done and reward < 0.2:  # penalize if reaching time limit but not accept
            reward = -1
        #print(self._seq, reward)
        #print(reward)
        #self.visualize()
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
        #print("reward", reward)
        #print("done", done)
        return self.feature(), reward, done, info

    def dist2items(self):
        # no need to -2 if no boundary
        total_ds = float(self._width + self._height)
        min_ds = np.ones(self.cookbook.n_kinds) * total_ds
        for y in reversed(range(self._height)):
            for x in range(self._width):
                if not (self.grid[x, y, :].any() or (x, y) == self.pos):
                    continue
                else:
                    thing = self.grid[x, y, :].argmax()
                    d = abs(x - self.pos[0]) + abs(y - self.pos[1])
                    #print(thing)
                    #print(self.cookbook.index.get(thing))
                    if min_ds[thing] > d:
                        min_ds[thing] = d
        # distance is zero if own the item
        for i, count in enumerate(self.inventory):
            if count > 0:
                min_ds[i] = 0
        return min_ds

    def closer_feature(self):
        ds = []
        # remember the current config
        current_ds = self.dist2items()
        current_grid = copy.deepcopy(self.grid)
        current_inventory = copy.deepcopy(self.inventory)
        current_workshop_outs = copy.deepcopy(self.workshop_outs)
        current_approaching = copy.deepcopy(self.approaching)
        #print("current_aproaching", list(current_approaching))
        current_pos = copy.deepcopy(self.pos)
        current_dir = copy.deepcopy(self.dir)
        for action in range(5):
            # take step
            self.step(action, no_eval=True)
            ds.append(current_ds-self.dist2items())
            # restore the current grid
            self.grid = copy.deepcopy(current_grid)
            self.inventory = copy.deepcopy(current_inventory)
            self.workshop_outs = copy.deepcopy(current_workshop_outs)
            self.approaching = copy.deepcopy(current_approaching)
            self.pos = copy.deepcopy(current_pos)
            self.dir = copy.deepcopy(current_dir)
        ds = np.asarray(ds)
        return np.transpose(ds, (1, 0))

    def feature(self):
        x, y = self.pos
        # position features
        pos_feats = np.asarray(self.pos).astype(np.float32)
        pos_feats[0] /= self._width
        pos_feats[1] /= self._height
        # direction features
        dir_features = np.zeros(4)
        dir_features[self.dir] = 1
        if self._use_gui:
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
            out_values = np.concatenate((self.inventory, pos_feats, dir_features))
            features = {0: out_img.astype(np.uint8), 1: out_values, 2: self.closer_feature(), 3: img}
        else:
            hw = int(self._window_width / 2)
            hh = int(self._window_height / 2)
            bhw = int((self._window_width * self._window_width) / 2)
            bhh = int((self._window_height * self._window_height) / 2)

            grid_feats = pad_slice(self.grid, (x-hw, x+hw+1), 
                    (y-hh, y+hh+1))
            out_grid = np.concatenate((grid_feats.ravel(),
                                       self.inventory,
                                       dir_features))
            features = {0: out_grid, 1: self.closer_feature()}
        return features

    def reset(self):
        self._state_visit_count = 0
        self._seq = []
        self._last_states = set(self.ba.get_initial_state())
        self.inventory = np.zeros(self.cookbook.n_kinds)
        self.workshop_outs = np.zeros(self.cookbook.n_kinds)
        self.approaching = []
        self.pos = copy.deepcopy(self._init_pos)
        self.dir = copy.deepcopy(self._init_dir)
        self.grid = copy.deepcopy(self._init_grid)
        if self._use_gui:
            self.gui.draw()
        return self.feature()

    def visualize(self):
        s = ''
        for y in reversed(range(self._height)):
            for x in range(self._width):
                if not (self.grid[x, y, :].any() or (x, y) == self.pos):
                    ch = ' '
                else:
                    thing = self.grid[x, y, :].argmax()
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
        for item, count in enumerate(self.inventory):
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


def sample_craft_env_each(args, width=7, height=7, env_data=None, env=None):
    #print()
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
        init_pos = random_free(grid, rand, width, height)
        init_dir = rand.randint(4)
        env_data = (grid, init_pos, init_dir)
    else:
        grid, init_pos, init_dir = env_data
    # return the env
    #print(env)
    if env is not None:
        env.load(env_data)
        return env
    else:
        return CraftWorldEnv(args.formula, args.recipe_path,
                             init_pos, init_dir, grid,
                             width=width, height=height,
                             use_gui=args.use_gui,
                             is_headless=args.is_headless,
                             prefix_reward_decay=args.prefix_reward_decay,
                             time_limit=args.num_steps,
                             target_fps=args.target_fps,
                             update_failed_trans_only=args.update_failed_trans_only)

def bfs(env, goal):
    #print("bfs")
    #print("goal")
    #print(env.cookbook.index.get(goal))
    #print()
    visited = set()
    queue = [(env.pos, [])]
    delts = env.del2direction.keys()
    while len(queue) > 0:
        curr, curr_seq = queue.pop(0)
        visited.add(curr)
        x, y = curr
        for d in delts:
            x_n, y_n = x+d[0], y+d[1]
            if not (x_n in range(env._width) and y_n in range(env._height)):
                continue
            if not env.grid[x_n, y_n, :].any():
                if (x_n, y_n) not in visited:
                    new_seq = curr_seq.copy()
                    new_seq.append(env.del2direction[d])
                    queue.append(((x_n, y_n), new_seq))
                continue
            thing = env.grid[x_n, y_n, :].argmax()
            #print(env.cookbook.index.get(goal))
            #print()
            if thing == goal:
                curr_seq.append(env.del2direction[d])
                curr_seq.append(int(CraftWorldEnv.Actions.use))
                #print(curr_seq)
                return True, curr_seq, env.del2direction[d]
    return False, [], None

def use_item(env, item, drop=True):
    #print(env.grid)
    item_name = env.cookbook.index.get(item)
    if drop:
        item_name = "recycle_" + item_name
    if 'none' in item_name:
        item_name = "recycle_" + env.cookbook.output2input[item_name]
    item_idx = env.cookbook.index[item_name]
    success, seq, last_action = bfs(env, item_idx)
    if success:
        return seq
    return []    

def get_items(env):
    things = []
    for i in range(env._height):
        for j in range(env._width):
            if env.grid[i, j, :].any():
                thing = env.grid[i, j, :].argmax()
                things.append(thing)
    return things

def gen_actions(env, max_n_seq, goal_only=True, n_tracks=1):
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
                items = [i for i in range(len(env.inventory)) if env.inventory[i] > 0]
                items += [i for i in range(len(env.workshop_outs)) if env.workshop_outs[i] > 0]
                items += get_items(env)
                items = list(filter(lambda x: x not in env.cookbook.workshop_indices, items))

                item = random.choice(items)
                drop = env.inventory[item] > 0
                #print([env.cookbook.index.get(i) for i in items])
                subtask = use_item(env, item, drop=drop)
                if len(subtask) > 1: 
                    last_waypoint_action = subtask[-2]
                
            subtask = subtask[:diff]
            actions += subtask

            #print(subtask)
            for j in range(len(subtask)):
                action = subtask[j]
                #print()
                #print("action", action)
                obs, reward, done, _ = env.step(action)

                #print(reward, done)
                if reward < 0.9:
                    found_len = step_j
                if done:
                    if (goal_only and reward > 0.9) or (not goal_only and reward > 0):
                        assert reward > 0.9
                        return True, actions
                        #seqs.append((actions, found_len))#TODO change back
                step_j += 1
                #print(env._formula)
                if reward < 0:
                    #print("violated")
                    violated = True
                    break
            if violated:
                break
    #if len(seqs) > n_tracks:#TODO change back
    #    seqs = list(sorted(seqs, key=lambda x: x[1])) #NOTE: not needed anymore?
    #    return True, seqs
    return False, []


def sample_craft_env(args, width=7, height=7, n_steps=1, k_path=5,
                     n_retries=5, env_data=None, train=False, check_action=True,
                     max_n_seq=50, goal_only=True, n_tracks=1):
    no_env = True; env = None; count = 0; actions = None
    # exclude the env that is true at beginning
    while no_env:
        env = sample_craft_env_each(args, width, height, env_data, env)
        grid = env.grid
        #print(grid)
        #exit()
        for i in range(k_path):  # try k_path random paths
            #<czw>
            env.reset()
            if i < 5:#test to make sure that no 
                action = i
                obs, reward, done, _ = env.step(action)
                no_env = reward > 0.5 #or (j == 0 and done)
                if no_env:
                    break
            #</czw>
            env.reset()
            for j in range(n_steps):
                action = np.random.choice(5)
                obs, reward, done, _ = env.step(action)
                # not sample an env if too easy or fail at first step
                no_env = reward > 0.5 #or (j == 0 and done)
                if no_env:
                    break
            if no_env:
                break
        if not no_env and check_action:  # filter out the env that doesn't have a solution
            env.reset()
            success, actions = gen_actions(env, max_n_seq=max_n_seq, goal_only=goal_only, n_tracks=1)
            if not success:
                no_env = True
        count += 1
        if count > n_retries:  # retry too many times
            print('Tried')
            break
    if no_env:
        env.should_skip = True
    env.reset()
    if not train and no_env:
        return None, []
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
    # args.recipe_path = ROOT_PATH + 'craft_recipes_basic.yaml'
    args.recipe_path = ROOT_PATH + 'craft_recipes_basic_switch.yaml'
    args.num_steps = 20
    args.target_fps = 60
    args.use_gui = False
    args.is_headless = True
    # args.formula = "( G ( tree ) )"#NOTE dummy formula
    args.formula = "( F ( tree ) )"
    args.prefix_reward_decay = 0.8
    args.update_failed_trans_only = False
    args.return_screen = False
    env, actions = sample_craft_env(args, n_steps=3, n_retries=5, train=True)
    if env is None: exit()
    env.ba.draw('tmp_images/ba.svg', show=False)
    '''
    while True:
        env.gui.draw(move_first=True)
        feature = env.feature()
        img = Image.fromarray(feature[0])
        img.save('tmp_images/feature.png')
    '''
