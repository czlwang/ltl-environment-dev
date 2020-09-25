import argparse
import numpy as np
import pickle
import worlds.craft_world as craft

from ltl2tree import ltl2tree, LTL_OPS
from spot2ba import Automaton


def get_args():
    parser = argparse.ArgumentParser(description='Compute stats of given formulas')
    parser.add_argument('--env_name', default='CharStream',
                        help='environment to test on: CharStream | Craft')
    parser.add_argument('--formula_pickle_prefix', default='./data/test_formula',
                        help='path to load the test formulas (default: ./data/test_formula)')
    parser.add_argument('--num_files', type=int, default=1,
                        help='number of pickle files to compute stats (default: 1)')
    args = parser.parse_args()
    return args


def load_formulas(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    formulas = [(f[0], Automaton(f[0]), f[1]) for f in data]  # use None for Buchi
    return formulas


def set_env_param(args):
    if args.env_name == 'CharStream':
        args.alphabets = ['a', 'b', 'c', 'd', 'e']
    elif args.env_name == 'Craft':
        args.recipe_path = 'worlds/craft_recipes_basic.yaml'
        args.alphabets = craft.get_alphabets(args.recipe_path)
        args.is_headless = True
        args.use_gui = True
        args.target_fps = None
    return args


def num_symbols(formulas):
    lengths = []
    for ltl, _, _ in formulas:
        length = len([s for s in ltl.split(' ') if s != '(' and s != ')' and s not in LTL_OPS])
        lengths.append(length)
    lengths = np.array(lengths)
    return np.mean(lengths), np.std(lengths)


def num_tree_nodes(formulas, args):
    n_nodes = []
    for ltl, _, _ in formulas:
        tree = ltl2tree(ltl, args.alphabets)
        n_nodes.append(tree.size)
    n_nodes = np.array(n_nodes)
    return np.mean(n_nodes), np.std(n_nodes)


def depth_tree(formulas, args):
    depths = []
    for ltl, _, _ in formulas:
        tree = ltl2tree(ltl, args.alphabets)
        depths.append(tree.depth)
    depths = np.array(depths)
    return np.mean(depths), np.std(depths)


def num_states(formulas):
    n_states = []
    for _, ba, _ in formulas:
        n_states.append(ba.n_states)
    n_states = np.array(n_states)
    return np.mean(n_states), np.std(n_states)


def main():
    args = get_args()
    args = set_env_param(args)
    formulas = []
    for n in range(args.num_files):
        filename = args.formula_pickle_prefix + '_' + str(n) + '.pickle'
        formulas.extend(load_formulas(filename))
    print('# of symbols: {}'.format(num_symbols(formulas)))
    print('# of tree nodes: {}'.format(num_tree_nodes(formulas, args)))
    print('depth of tree: {}'.format(depth_tree(formulas, args)))
    print('# of Buchi Automata states: {}'.format(num_states(formulas)))


if __name__ == '__main__':
    main()
