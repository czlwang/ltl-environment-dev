# LTL Environment Dev

Environment grid world for [Encoding formulas as deep networks: Reinforcement learning for zero-shot execution of LTL formulas](https://arxiv.org/pdf/2006.01110.pdf), based on the Craft-world from [Andreas et al. (2017)](https://arxiv.org/pdf/1611.01796.pdf)

## Structure
* `algo/` contains the code to train a model using Advantage Actor Critic
* `language/` contains the code to generate a dataset of (sentence, formula, environment) examples 
  * usage: `language/dataset.py --n_sentence 10 --dataset_path examples_out`
* `worlds/craft_world.py` contains all the rules of the craft world environment as well as it's GUI
* `spot2ba.py` contains the hooks into [Spot](https://spot.lrde.epita.fr/) needed to maintain the FSA
* `ltl2tree.py` contains the code that specifies the tree structure of the planner model. This tree is formed according to the parse of the given LTL formula.
