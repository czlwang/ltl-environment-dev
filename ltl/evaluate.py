import argparse
import numpy as np
import pickle
import torch
import worlds.craft_world as craft
import utils

from algo.a2c_acktr import A2C_ACKTRTrainer
from envs import make_single_env
from ltl2tree import ltl2tree, ltl2onehot
from PIL import Image
from spot2ba import Automaton


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate RL with LTL')
    parser.add_argument('--env_name', default='CharStream',
                        help='environment to test on: CharStream | Craft')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='number of forward steps in A2C (default: 10)')
    parser.add_argument('--save_model_dir', default='models/',
                        help='path of the folder to the saved model (default: models/)')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='evaluate baseline model')
    parser.add_argument('--random', action='store_true', default=False,
                        help='evaluate random output')
    parser.add_argument('--no_time', action='store_true', default=False,
                        help='evaluate no time dependency')
    parser.add_argument('--test_formula_pickle', default='./data/test_formula.pickle',
                        help='path to load the test formulas (default: ./data/test_formula.pickle)')
    parser.add_argument('--num_updates', type=int, default=10000,
                        help='number of updates logged (default: 100)')
    parser.add_argument('--result_file', default='./results/final.csv',
                        help='path to write the result file (default: ./results/final.csv)')
    parser.add_argument('--load_env_data', action='store_true', default=False,
                        help='load environment data')
    parser.add_argument('--env_data_path', default='./data/env.pickle',
                        help='path to load environment data (default: ./data/env.pickle)')
    parser.add_argument('--min_updates', type=int, default=0,
                        help='starting number of updates to evaluate (default: 0)')
    parser.add_argument('--lang_emb', action='store_true', default=False,
                        help='train the language embedding baseline')
    parser.add_argument('--model_path', default='',
                        help='path to load trained model')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # args from training
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | acktr | sac')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='dimensions of the RNN hidden layers.')
    parser.add_argument('--rnn_depth', type=int, default=1,
                        help='number of layers in the stacked RNN.')
    parser.add_argument('--output_state_size', type=int, default=32,
                        help='dimensions of the output interpretable state vector.')
    parser.add_argument('--lang_emb_size', type=int, default=32,
                        help='embedding size of the ltl formula (default: 32)')
    parser.add_argument('--image_emb_size', type=int, default=64,
                        help='embedding size of the input image (default: 64)')
    args = parser.parse_args()
    # cuda setting
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(utils.choose_gpu() if args.cuda else "cpu")
    args.device = device
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
    args.prefix_reward_decay = 0.03
    return args


def get_agent(args):
    # load test env
    data = []
    if args.load_env_data:
        with open(args.env_data_path, 'rb') as f:
            data = pickle.load(f)
    # test for each formula
    if len(data) > 0:
        env = make_single_env(args, data[0])
    else:
        env = make_single_env(args, None)
    # load model
    ltl_tree = ltl2tree(args.formula, args.alphabets, args.baseline)
    args.observation_space = env.observation_space
    args.action_space = env.action_space
    agent = A2C_ACKTRTrainer(ltl_tree, args.alphabets, args)
    return agent, args


def test(args, formulas, agent, random=False):
    # load test env
    data = []
    if args.load_env_data:
        with open(args.env_data_path, 'rb') as f:
            data = pickle.load(f)
    # test for each formula
    if len(data) > 0:
        env = make_single_env(args, data[0])
    else:
        env = make_single_env(args, None)
    if not random:
        agent.actor_critic.load_state_dict(torch.load(args.model_path)[0])
    n_successes = 0
    final_steps = np.zeros(args.num_steps)
    for i, i_formula in enumerate(formulas):
        formula, ba, _ = i_formula
        env.close(); del env
        args.formula = formula
        if len(data) > 0:
            env = make_single_env(args, data[i])
        else:
            env = make_single_env(args, None)
        if args.load_env_data and len(data) > 0:
            env.load(data[i])
        ltl_tree = ltl2tree(args.formula, args.alphabets, args.baseline)
        if args.lang_emb:
            agent.update_formula(ltl_tree, ltl2onehot(args.formula, args.alphabets))
        else:
            agent.update_formula(ltl_tree)
        agent.actor_critic.eval()
        with torch.no_grad():
            agent.actor_critic.reset()
            obs = env.reset()
            done = False; accumulated_reward = 0
            success = False
            final_step = 0
            for step in range(args.num_steps):
                # Sample actions
                if type(obs) is dict:
                    test_obs = []
                    for _, s in obs.items():
                        test_obs.append(torch.FloatTensor(s))
                        test_obs[-1] = test_obs[-1].to(args.device)
                    test_obs = tuple(test_obs)
                else:
                    test_obs = torch.FloatTensor(obs)
                    test_obs = test_obs.to(args.device)
                with torch.no_grad():
                    mask = torch.FloatTensor([1.0])
                    mask = mask.to(args.device)
                    _, action, _ = agent.actor_critic.act(test_obs, mask,
                        deterministic=False, no_hidden=args.no_time)
                # Observation, reward and next obs
                obs, reward, done, infos = env.step(action[0])
                accumulated_reward += reward
                if done:
                    final_steps[step] += 1
                    break
        if step == args.num_steps - 1 and accumulated_reward > 1:
            success = True
            n_successes += 1
        '''
        if args.env_name == 'CharStream':
            print('{},{},{},{},{}'.format(i, formula, int(env.ba.num_accept_str(args.num_steps)), step, success))
        else:
            print('{},{},{},{}'.format(i, formula, step, success))
        '''
    env.close(); del env  # close the env so no EOF error
    del agent
    return n_successes, final_steps


def evaluate_folder(args):
    result_file = open(args.result_file, 'a')
    formulas = load_formulas(args.test_formula_pickle)
    args.formula, _, _ = formulas[0]
    agent, args = get_agent(args)
    for n in range(args.num_updates):
        if n < args.min_updates:
            continue
        print('Evaluate update {}'.format(n))
        args.model_path = args.save_model_dir + '/model_' + str(n) + '.pt'
        args.formula, _, _ = formulas[0]
        n_successes, _ = test(args, formulas, agent)
        accuracy = n_successes / len(formulas)
        print(' Accuracy:', accuracy)
        result_file.write('{},{}\n'.format(n, accuracy))
        result_file.flush()
    result_file.close()


def evalute_model(args):
    formulas = load_formulas(args.test_formula_pickle)
    args.formula, _, _ = formulas[0]
    agent, args = get_agent(args)
    args.formula, _, _ = formulas[0]
    n_successes, final_steps = test(args, formulas, agent)
    accuracy = n_successes / len(formulas)
    print(' Accuracy:', accuracy)
    print(' Number of steps distribution')
    for i in range(args.num_steps):
        print(' {}: {}'.format(i, final_steps[i]))


def baseline_random_steps(args):
    formulas = load_formulas(args.test_formula_pickle)
    args.formula, _, _ = formulas[0]
    agent, args = get_agent(args)
    n_successes, final_steps = test(args, formulas, agent, random=True)
    accuracy = n_successes / len(formulas)
    print('Accuracy:', accuracy)
    print(' Number of steps distribution')
    for i in range(args.num_steps):
        print(' {}: {}'.format(i, final_steps[i]))


def main():
    args = get_args()
    args = set_env_param(args)
    if args.random:
        baseline_random_steps(args)
    elif args.model_path != '':
        evalute_model(args)
    else:
        evaluate_folder(args)


if __name__ == '__main__':
    main()
