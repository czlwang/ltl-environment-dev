import gym
import numpy as np
import random

from gym import spaces
from gym.utils import seeding
from ltl.spot2ba import Automaton
from ltl.ltl_sampler import ltl_sampler


def transition_to_onehot(trans, alphabets):
    seq = []
    alphabet2id = dict()
    idx = 0
    for alphabet in alphabets:
        alphabet2id[alphabet] = idx
        idx += 1
    for tran in trans:
        v = np.zeros(len(alphabets))
        for s in tran:
            v[alphabet2id[s]] = 1
        seq.append(v)
    return np.array(seq)


def onehot_to_transition(onehots, alphabets):
    seq = []
    id2alphabet = dict()
    idx = 0
    for alphabet in alphabets:
        id2alphabet[idx] = alphabet
        idx += 1
    for onehot in onehots:
        v = set()
        for idx, val in enumerate(onehot):
            if val > 0:
                v.add(id2alphabet[idx])
        seq.append(v)
    return seq


class CharStreamEnv(gym.Env):
    def __init__(self, formula, alphabets,
                 prefix_reward_decay=1., time_limit=10):
        self.action_space = spaces.MultiBinary(len(alphabets))
        self.observation_space = spaces.MultiBinary(len(alphabets))
        self.prefix_reward_decay = prefix_reward_decay
        self.time_limit = time_limit
        self.seed()
        # convert the ltl formula to a Buchi automaton
        self.ba = Automaton(formula, alphabets)
        self._formula = formula
        self._alphabets = alphabets
        self._seq = []
        self._last_states = set(self.ba.get_initial_state())
        self._state_visit_count = 0
        # start the first game
        self.should_skip = False
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_data(self):
        return None

    def load(self):
        pass

    def step(self, action, random=False):
        if random:
            action = np.random.choice([0., 1.], len(self._alphabets))
        assert self.action_space.contains(action)
        self._seq.append(action)
        done = len(self._seq) >= self.time_limit  # done when reaching time limit
        trans = onehot_to_transition(self._seq, self._alphabets)
        is_prefix, dist_to_accept, last_states, failed_trans = \
                self.ba.is_prefix([trans[-1]], self._last_states)
        is_accpet = dist_to_accept < 0.1
        if is_accpet:  # not done even if it is in accept state
            reward = 1
            self._last_states = last_states
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
            done = True  # stay at done if the env doesn't reset
        return np.array(action, dtype=np.int8), reward, done, {'failed_trans': failed_trans}

    def reset(self):
        self._seq = []
        self._last_states = set(self.ba.get_initial_state())
        return np.array([0 for _ in self._alphabets], dtype=np.int8)


if __name__ == '__main__':
    # sample a ltl formula
    alphabets = ['a', 'b', 'c']
    ltls = ltl_sampler(alphabets, n_samples=1)
    ltl, ba, _ = ltls[0]
    print('LTL formula:', ltl)
    ba.draw('tmp_images/ba.svg', show=False)
    states, trans = ba.gen_sequence()
    print('min sequence length:', ba.len_min_accepting_run)
    print('avg sequence length:', ba.len_avg_accepting_run)
    print('alphabets', ba._alphabets)
    print('states', states)
    print('trans', trans)
    one_hots = transition_to_onehot(trans, alphabets=ba._alphabets)
    print('one-hot:\n', one_hots)
    # run char stream environment
    env = CharStreamEnv(ltl, ba._alphabets)
    for one_hot in one_hots:
        print(one_hot, env.step(one_hot))
