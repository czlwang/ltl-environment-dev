import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ltl.algo.kfac import KFACOptimizer
from ltl.model import LTLActorCritic


class A2C_ACKTRTrainer(object):
    def __init__(self, ltl_tree, symbols, args, acktr=False):
        actor_critic = LTLActorCritic(ltl_tree, symbols, args)
        actor_critic.to(args.device)

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        self.max_grad_norm = args.max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            if args.env_name == 'Craft':
                #self.optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr, weight_decay=0.0005)
                self.optimizer = optim.RMSprop(
                    actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
            else:
                self.optimizer = optim.RMSprop(
                    actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)

    def update_formula(self, ltl_tree, ltl_onehot=None):
        self.actor_critic.update_formula(ltl_tree, ltl_onehot)
        # reset optimizer when switching to new formula
        if self.acktr:
            self.optimizer.steps = 0

    def update(self, rollouts):
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values = []
        action_log_probs = []
        dist_entropy = []
        self.actor_critic.reset()  # reset to set the starting hidden states correct
        for i in range(num_steps):
            i_values, i_action_log_probs, i_dist_entropy = self.actor_critic.evaluate_actions(
                rollouts.get_obs(i),
                rollouts.masks[i],
                rollouts.actions[i])
            values.append(i_values)
            action_log_probs.append(i_action_log_probs)
            dist_entropy.append(i_dist_entropy)

        values = torch.stack(values)
        action_log_probs = torch.stack(action_log_probs)
        dist_entropy = torch.stack(dist_entropy).mean()

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda(values.device)

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
