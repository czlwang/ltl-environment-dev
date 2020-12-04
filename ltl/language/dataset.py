import argparse
import json
import numpy as np
import random

from collections import defaultdict
from ltl.language.generator import SentenceGrammar
from ltl.ltl2tree import replace_symbols
from ltl.worlds.craft_world import sample_craft_env


def gen_env(args, formula, n_trials=3):
    '''
        returns a list of n environments with k tracks each
    '''
    args.formula = formula
    trial = 0
    envs, action_list = [], []
    while trial < n_trials:
        env, actions = sample_craft_env(args, max_n_seq=100, goal_only=True, k_path=5, height=7, width=7, n_tracks=1, for_dataset=True)
        if env is None:
            trial += 1
            success = False; actions = []
            continue
        else:
            success = True
            envs.append(env)
            action_list.append(actions)
        trial += 1

    if len(envs) < args.n_envs:
        success = False
    return success, envs, action_list


def filter_formula_by_binding(formula):
    # Filter out the formula that binds `|` first
    tokens = formula.split(' ')
    parans = []
    for token in tokens:
        if token == '(':
            parans.append(token)
        elif token == ')':
            if parans[-1] != '(':
                parans.pop()
            parans.pop()
        elif token == '&':
            parans.append(token)
        elif token == '|':
            if '&' in parans:
                return True
            parans.append(token)
    return False


def gen_dataset(args, n, max_length=15, count_per_sent=1):
    grammar = SentenceGrammar(args.recipe_path)
    dataset = []; sent_count = defaultdict(int)
    while len(dataset) < n:
        sentence, formula = grammar.gen_sentence(n=1)[0]
        symbols = formula.split(" ")
        symbols = [s for s in symbols if s not in ["(", ")"]]
        if len(symbols) > 15:#TODO get rid of this
            continue
        if sent_count[sentence] > count_per_sent:  # skip sentences that are sampled many times
            continue
        if filter_formula_by_binding(formula):  # prefer `((a & b) | c)` over `(a & (b | c))`
            continue
        tokens = [token for token in formula.split(' ') if token not in ['(', ')']]
        if len(tokens) > max_length:
            continue
        ltl = replace_symbols(formula, env_name='Craft')
        args.formula = ltl
        success, envs, actions = gen_env(args, ltl, args.n_trials)
        if not success:
            print('Skip:', sentence)
            continue

        print('Sentence:', sentence)
        print('  LTL:', formula)
        print('  Replaced LTL:', ltl)
        #print('  Ground-truth Actions:', actions, ', Length:', first_accepts)
        sent_count[sentence] += 1
        dataset.append((sentence, formula, ltl, envs, actions))
    return dataset


def one_hot2index(env, init_grid):
    grid = np.zeros((env._height, env._width), dtype=np.int8)
    for y in range(env._height):
        for x in range(env._width):
            if init_grid[x, y, :].any():
                thing = init_grid[x, y, :].argmax()
                grid[x, y] = thing
    return grid


def save_dataset_json(dataset, args):
    json_path = args.dataset_path
    output = {'data': []}
    for sentence, ltl, rewritten_ltl, envs, actions in dataset:
        actions_d, env_d = [], []

        for i in range(len(envs)):
            actions_i = actions[i]
            #actions_i, first_accepts = map(list, zip(*actions_i))
            #first_accepts = list(first_accepts)
            
            actions_i, first_accepts = list(map(list, zip(actions_i)))
            first_accepts = list(first_accepts)

            env = envs[i]
            init_grid, init_pos, init_dir = env.get_data()
            grid = one_hot2index(env, init_grid)
            actions_d.append({"steps": actions_i, "first_accept": first_accepts})
            env_i = {'init_grid': grid.tolist(), 'init_pos': list(init_pos), 'init_dir': init_dir} #, 'init_grid_one_hot': init_grid.tolist()},
            env_d.append(env_i)

        # prepare grid
        data = {'sentence': sentence,
                'formula': ltl,
                'rewritten_formula': rewritten_ltl,
                'envs': env_d,
                'actions': actions_d}
        output['data'].append(data)
    with open(json_path, 'w') as json_file:
        json.dump(output, json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate language dataset for the craft world')
    parser.add_argument('--n_sentence', type=int, default=10,
                        help='number of sentences to generate for the dataset (default: 10)')
    parser.add_argument('--dataset_path', default='data/sentences.json',
                        help='path to save the dataset')
    parser.add_argument('--num_steps', type=int, default=15,
                        help='number of steps to take in the environment (default: 15)')
    args = parser.parse_args()
    # other predefined params
    args.recipe_path = 'worlds/craft_recipes_basic.yaml'
    args.prefix_reward_decay = 0.8
    args.target_fps = 60
    args.use_gui = False
    args.is_headless = True
    args.update_failed_trans_only = False
    args.n_trials = 20
    args.n_envs = 10
    args.num_steps = 20

    print(args)
    # generate dataset
    dataset = gen_dataset(args, n=args.n_sentence)
    save_dataset_json(dataset, args)

