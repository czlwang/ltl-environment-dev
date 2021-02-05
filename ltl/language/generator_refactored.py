import random
from ltl.spot2ba import Automaton
import ltl.worlds.craft_world as craft
import re

from collections import defaultdict
 
# S -> Safety | Guarantee | Obligation | Recurrence | Persistence | Reactivity | Conditional
# TODO: add `grass` and `toolshed` back
GRAMMAR = """
    BinOp -> 'and' | 'or'
    UOp -> 'do not' | 'you should not'
    Not -> 'not'
    Item -> 'apple' | 'orange' | 'pear'
    Landmark -> 'flag' | 'house' | 'tree'
    Color -> 'blue'
    Condition -> Color
    Predicate -> 'be around the' Landmark | 'be near the' Landmark | 'go to the' Landmark | 'hold the' Item | 'take the' Item | 'possess the' Item
    pp -> Predicate | Predicate BinOp Predicate
    p -> Predicate | Predicate BinOp Predicate | UOp pp
    S -> Safety | Guarantee | Obligation | Recurrence | Persistence | Reactivity
    SPrefix -> 'always' | 'at all times,'
    SSuffix -> 'forever' | 'at all times' | 'all the time'
    Safety -> SPrefix p | p SSuffix | Safety BinOp Safety
    GPrefix -> 'eventually' | 'at some point'
    NotPredicate -> UOp Predicate
    Guarantee -> GPrefix p | 'guarantee that you will' Predicate | 'guarantee that you' NotPredicate | Guarantee BinOp Guarantee
    Obligation -> Safety BinOp Guarantee | Obligation BinOp Safety | Obligation BinOp Guarantee
    Recurrence -> 'eventually,' p 'and do this repeatedly' | Recurrence BinOp Recurrence
    Persistence -> 'at some point, start to' p 'and keep doing it' | Persistence BinOp Persistence
    Reactivity -> Recurrence BinOp Persistence | Reactivity BinOp Recurrence | Reactivity BinOp Persistence
    Conditional -> 'while' Condition p | p 'until' Condition
"""

LTL_GRAMMAR = """
    BinOp -> '&' | '|'
    UOp -> '!' | '!'
    Not -> '!'
    Item -> 'apple' | 'orange' | 'pear'
    Landmark -> 'flag' | 'house' | 'tree'
    Color -> 'blue'
    Condition -> Color
    Predicate -> Landmark | Landmark | Landmark | Item | Item | Item
    pp -> Predicate | Predicate BinOp Predicate
    p -> Predicate | Predicate BinOp Predicate | UOp pp
    S -> Safety | Guarantee | Obligation | Recurrence | Persistence | Reactivity
    SPrefix -> 'F' | 'F'
    SSuffix -> 'F' | 'F' | 'F'
    Safety -> SPrefix '(' p ')' | SSuffix '(' p ')' | '(' Safety ')' BinOp '(' Safety ')'
    GPrefix -> 'G' | 'G'
    NotPredicate -> UOp Predicate
    Guarantee -> GPrefix '(' p ')' | 'G (' Predicate ')' | 'G (' NotPredicate ')' | '(' Guarantee ')' BinOp '(' Guarantee ')'
    Obligation -> '(' Safety ')' BinOp '(' Guarantee ')' | '(' Obligation ')' BinOp '(' Safety ')' | '(' Obligation ')' BinOp '(' Guarantee ')'
    Recurrence -> 'G F (' p ')' | '(' Recurrence ')' BinOp '(' Recurrence ')'
    Persistence -> 'F G (' p ')' | '(' Persistence ')' BinOp '(' Persistence ')'
    Reactivity -> '(' Recurrence ')' BinOp '(' Persistence ')' | '(' Reactivity ')' BinOp '(' Recurrence ')' | '(' Reactivity ')' BinOp '(' Persistence ')'
    Conditional -> '(' p ') U ! (' Condition ')' | '(' p ') U (' Condition ')'
"""

class TreeNode:
    def __init__(self, name, grammar, prod):
        # The name of the variable
        self.name = name

        self.grammar = grammar
        self.prod = prod
        self.choice = None
        self.nodes = {}
    
    def get_depth(self):
        return max([0] + [n.get_depth() for n in self.nodes.values()]) + 1
    
    # This removes the underscore from the name of the symbol if it has one
    def get_symbol(self):
        return self.name.split('_')[0]
    
    # Splits the grammar up into its symbols and strings
    def split_cfg(self, l):
        inside_quote = False
        cur_string = ""
        new_list = []
        for elem in l.split():
            if ("'" in elem):
                if (elem.count("'") == 2):
                    new_list.append(elem[1:-1])
                    continue
                elif (inside_quote):
                    new_list.append(cur_string + " " + elem[:-1])
                else:
                    cur_string = elem[1:]
                inside_quote = not inside_quote
                continue

            if (inside_quote):
                cur_string += " " + elem
            else:
                new_list.append(elem)
        return new_list
    
    def create_weights(self, pcount, excludes=[], cfactor=0.25):
        weights = []
        for i, p in enumerate(self.prod[self.get_symbol()]):
            skip = False
            for token in excludes:
                if token in p:
                    weights.append(0.01)
                    skip = True
                    break
            if skip:
                continue
            elif p in pcount:
                weights.append(cfactor ** (pcount[p]))
            else:
                weights.append(1.0)
        return weights

    def weighted_choice(self, weights):
        rnd = random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i
    
    # TODO: Excludes is currently not being used
    # TODO: Update pcount

    # Creates a tree using the value passed in
    def create_tree(self, pcount=defaultdict(int), excludes=[], sim_node=None):
        # Keeps track of symbols seen so far to properly label them
        symbols = defaultdict(int)
        
        # We only need the weights if we aren't following a path
        weights = []
        if (not sim_node):
            weights = self.create_weights(pcount, excludes)
        
        self.value = []
        rand_prod = ''
        if (not sim_node):
            self.choice = self.weighted_choice(weights)
            rand_prod = self.prod[self.get_symbol()][self.choice]
            pcount[rand_prod] += 1
            self.value = self.split_cfg(rand_prod)
        else:
            self.value = self.split_cfg(self.prod[self.get_symbol()][sim_node.choice])
        for i, v in enumerate(self.value):
            if (v.split('_')[0] in self.prod.keys()):
                # Give the node a unique name if it doesn't have one
                node_name = v
                if (not '_' in v):
                    node_name += '_' + str(symbols[v])
                    symbols[v] += 1

                if (not sim_node):
                    node = TreeNode(node_name, self.grammar, self.prod)
                    
                    # Save the choice being made so this path can be traversed later
                    self.nodes[node_name] = node
                    
                    node.create_tree(pcount)
                    self.value[i] = node
                else:
                    next_node = sim_node.nodes[node_name]
                    node = TreeNode(node_name, self.grammar, self.prod)
                    node.create_tree(sim_node=next_node)
                    self.value[i] = node

        # Backtracking: clear the modification to pcount    
        pcount[rand_prod] -= 1

    # Returns the sentence
    def __str__(self):
        string = ''
        for v in self.value:
            string += str(v) + ' '
        return string[:-1]

class SentenceGrammar(object):
    def __init__(self, recipe_path):
        self.grammar = self.create_grammar(recipe_path, GRAMMAR)
        self.ltl_grammar = self.create_grammar(recipe_path, LTL_GRAMMAR)
        self._prod = self.parse_grammar(self.grammar)
        self._ltl_prod = self.parse_grammar(self.ltl_grammar)

    def create_grammar(self, recipe_path, input_grammar):
        rules = filter(lambda x: x != '', input_grammar.split('\n'))
        cookbook = craft.Cookbook(recipe_path)
        grammar = ''
        for rule in rules:
            line = ''
            if (rule.split()[0] == 'Item'):
                line = '    Item -> '
                for primitive in cookbook.original_recipes['primitives']:
                    line += '\'' + primitive + '\' | '
                line = line[:-3]
            elif (rule.split()[0] == 'Landmark'):
                line = '    Landmark -> '
                for landmark in cookbook.original_recipes['environment']:
                    # TODO: This is a very hacky way to get the landmarks
                    if (not '_' in landmark):
                        line += '\'' + landmark + '\' | '
                line = line[:-3]
            else:
                line = rule
            grammar += line + '\n'
        grammar = grammar[:-1]
        return grammar

    def parse_grammar(self, grammar):
        rules = filter(lambda x: x != '', grammar.split('\n'))
        prod = defaultdict(list)
        for rule in rules:
            rule = rule.strip().split(' -> ')
            lhs = rule[0]; rhs = rule[1]
            prods = rhs.split(' | ')
            for p in prods:
                prod[lhs].append(p)
        return prod

    def _weighted_choice(self, weights):
        rnd = random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i
    
    def split_cfg(self, l):
        inside_quote = False
        cur_string = ""
        new_list = []
        for elem in l:
            if ("'" in elem):
                if (elem.count("'") == 2):
                    new_list.append(elem[1:-1])
                    continue
                elif (inside_quote):
                    new_list.append(cur_string + " " + elem[:-1])
                else:
                    cur_string = elem[1:]
                inside_quote = not inside_quote
                continue

            if (inside_quote):
                cur_string += " " + elem
            else:
                new_list.append(elem)
        return new_list

    def gen_random(self, symbol, prod, cfactor=0.25, pcount=defaultdict(int), excludes=None, negate=False, choices=[], is_formula=False, depth=0):
        output = ''; weights = [];
        if (not is_formula):
            if excludes is None:
                excludes = []
            for i, p in enumerate(self._prod[symbol]):
                skip = False
                for token in excludes:
                    if token in p:
                        weights.append(0.01)
                        skip = True
                        break
                if skip:
                    continue
                elif p in pcount:
                    weights.append(cfactor ** (pcount[p]))
                else:
                    weights.append(1.0)
        
        # sample for a production
        if (not is_formula):
            selection = self._weighted_choice(weights)
            choices[depth].append((selection, symbol))
            rand_prod = prod[symbol][selection]
            pcount[rand_prod] += 1
        else:
            # Find the correct symbol at the correct depth
            elem = None
            for e in choices[depth]:
                if (e[1] == symbol):
                    elem = e
                    break
            rand_prod = prod[symbol][e[0]]
            choices[depth].remove(e)

        # Split rand_prod up into it's individual elements
        elems = self.split_cfg(rand_prod.split())
        for e in elems:
            if e in self._prod.keys():
                i_output = self.gen_random(e, prod, cfactor=cfactor, pcount=pcount, excludes=excludes, negate=negate, choices=choices, is_formula=is_formula, depth=depth+1)
                output += i_output
            else:
                output += e + ' '
        
        # backtracking: clear the modification to pcount
        if (not is_formula):
            pcount[rand_prod] -= 1
            exclude = None

        output = output.replace("to do not", "to not")
        output = output.replace("to you should not", "to not")
        output = output.replace("you do not be", "you not be")
        return output

    def gen_sentence(self, n=1):
        sentences = []
        for i in range(n):
            choices = defaultdict(list)
            node = TreeNode('S', self.grammar, self._prod)
            node.create_tree()
            ltl_node = TreeNode('S', self.ltl_grammar, self._ltl_prod)
            ltl_node.create_tree(sim_node=node)
            sentence = str(node) 
            formula = str(ltl_node)
            sentences.append((sentence, formula))
        return sentences

def gen_ltl_example():
    grammar = SentenceGrammar('/storage/dsleeper/RL_Parser/ltl/ltl/worlds/craft_recipes_basic_color.yaml')
    choices = defaultdict(list)
    node = TreeNode('S', grammar.grammar, grammar._prod)
    node.create_tree()
    ltl_node = TreeNode('S', grammar.ltl_grammar, grammar._ltl_prod)
    ltl_node.create_tree(sim_node=node)
    return str(ltl_node)

if __name__ == '__main__':
    grammar = SentenceGrammar('/storage/dsleeper/RL_Parser/ltl/ltl/worlds/craft_recipes_basic_color.yaml')
    for i in range(10):
        node = TreeNode('S', grammar.grammar, grammar._prod)
        node.create_tree()
        print(str(node))
        ltl_node = TreeNode('S', grammar.ltl_grammar, grammar._ltl_prod)
        ltl_node.create_tree(sim_node=node)
        print(str(ltl_node))
        

    '''
    for sentence, formula in grammar.gen_sentence(n=10):
        print('Sentence:', sentence)
        print('  LTL:', formula)
        # alphabets = ['boundary', 'C_boundary', 'tree', 'C_tree', 'house', 'C_house', 'flag', 'C_flag', 'orange', 'C_orange','apple', 'C_apple', 'pear', 'C_pear']
        # alphabets = ['boundary', 'C_boundary', 'tree', 'C_tree', 'workbench', 'C_workbench', 'factory', 'C_factory', 'iron', 'C_iron', 'gold', 'C_gold', 'gem', 'C_gem', 'copper', 'C_copper']
        alphabets = ['boundary', 'C_boundary', 'tree', 'C_tree', 'workbench', 'C_workbench', 'factory', 'C_factory', 'iron', 'C_iron', 'gold', 'C_gold', 'gem', 'C_gem', 'blue_on', 'blue_off']
        Automaton(formula, alphabets, add_flexible_state=False) 
    '''

