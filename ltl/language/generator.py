import random
from ltl.spot2ba import Automaton
import ltl.worlds.craft_world as craft

from collections import defaultdict


# TODO: add `grass` and `toolshed` back
GRAMMAR = """
    BinOp -> 'and' | 'or'
    UOp -> 'do not' | 'you should not'
    Not -> 'not'
    Item -> 'apple' | 'orange' | 'pear'
    Landmark -> 'flag' | 'house' | 'tree'
    Predicate -> 'be around the' Landmark | 'be near the' Landmark | 'go to the' Landmark | 'hold the' Item | 'take the' Item | 'possess the' Item
    p -> Predicate | UOp Predicate | Predicate BinOp Predicate | UOp p
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
"""


#GRAMMAR = """
#    BinOp -> 'and' | 'or'
#    UOp -> 'do not' | 'avoid'
#    Not -> 'not'
#    Item -> 'apple' | 'orange' | 'pear'
#    Landmark -> 'flag' | 'house' | 'tree'
#    Predicate -> 'be around' Landmark | 'be near' Landmark | 'hold' Item | 'take' Item
#    p -> Predicate | UOp Predicate | Predicate BinOp Predicate | UOp p
#    S -> Safety | Guarantee | Obligation | Recurrence | Persistence | Reactivity
#    SPrefix -> 'always'
#    SSuffix -> 'forever'
#    Safety -> SPrefix p | p SSuffix | Safety BinOp Safety
#    GPrefix -> 'eventually' | 'at some point'
#    Guarantee -> GPrefix p | 'make' Predicate 'happen' | 'make' Predicate Not 'happen' | Guarantee BinOp Guarantee
#    Obligation -> Safety BinOp Guarantee | Obligation BinOp Safety | Obligation BinOp Guarantee
#    Recurrence -> 'at some point,' p 'for a while' | Recurrence BinOp Recurrence
#    Persistence -> 'at some point, start' p 'and keep doing it' | Persistence BinOp Persistence
#    Reactivity -> Recurrence BinOp Persistence | Reactivity BinOp Recurrence | Reactivity BinOp Persistence
#"""


CLASS_LTL_PREFIX = {
        'Safety': 'G ',
        'Guarantee': 'F ',
        'Recurrence': 'G F ',
        'Persistence': 'F G '
        }


class SentenceGrammar(object):
    def __init__(self, recipe_path):
        self._prod = defaultdict(list)
        self.grammar = ''
        self.create_grammar(recipe_path)
        self.parse_grammar()

    def create_grammar(self, recipe_path):
        rules = filter(lambda x: x != '', GRAMMAR.split('\n'))
        cookbook = craft.Cookbook(recipe_path)
        for rule in rules:
            line = ''
            if (rule.split()[0] == 'Item'):
                line = '    Item -> '
                for primitive in cookbook.original_recipes['primitives']:
                    line += primitive + ' | '
                line = line[:-3]
            elif (rule.split()[0] == 'Landmark'):
                line = '    Landmark -> '
                for landmark in cookbook.original_recipes['environment']:
                    # TODO: This is a very hacky way to get the landmarks
                    if (not '_' in landmark):
                        line += landmark + ' | '
                line = line[:-3]
            else:
                line = rule
            self.grammar += line + '\n'
        self.grammar = self.grammar[:-1]

    def parse_grammar(self):
        rules = filter(lambda x: x != '', self.grammar.split('\n'))
        for rule in rules:
            rule = rule.strip().split(' -> ')
            lhs = rule[0]; rhs = rule[1]
            prods = rhs.split(' | ')
            for prod in prods:
                self._prod[lhs].append(prod)

    def gen_single_prod(self, prod, cfactor=0.25, pcount=defaultdict(int), excludes=None, negate=False):
        if '\'' in prod:
            tmp_tokens = filter(lambda x: x != '', prod.split('\''))
            # fix for Predicate Not, TODO: find a better fix
            tokens = []
            for token in tmp_tokens:
                if 'Not' in token:
                    tokens.extend(token.strip().split(' '))
                else:
                    tokens.append(token)
        else:
            tokens = filter(lambda x: x != '', prod.split(' '))
        out = []; formula = []; need_brackets = False
        if excludes is None:
            excludes = []
        should_negate = 'UOp' in prod or 'Not' in prod
        if should_negate:  # avoid double-negate if there is one negation
            excludes.extend(['UOp', 'Not'])
        for token in tokens:
            token = token.strip()
            if token in self._prod.keys():
                sentence, formula_part = self.gen_random(token, \
                        cfactor=cfactor, pcount=pcount, excludes=excludes, negate=should_negate)
                if token in ['Item', 'Landmark']:
                    formula.append('( ' + ''.join(sentence.split(' ')) + ' )')
                    excludes.append(sentence)
                elif sentence == 'and':
                    formula.append('&')
                    need_brackets = True
                elif sentence == 'or':
                    formula.append('|')
                    need_brackets = True
                elif token in ['UOp', 'Not']:
                    formula.append('!')
                    need_brackets = True
                    if token == 'Not':  # swap predicate and not
                        formula[-1] = formula[-2]; formula[-2] = '!'
                elif len(formula_part) > 0:
                    formula.append(formula_part)
                out.append(sentence)
            else:
                out.append(token)
        excludes = None
        # combine formulas
        if len(formula) > 0:
            formula = ' '.join(formula)
            if need_brackets:
                formula = '( ' + formula + ' )'
        else:
            formula = ''
        return ' '.join(out), formula

    def _weighted_choice(self, weights):
        rnd = random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i

    def gen_random(self, symbol, cfactor=0.25, pcount=defaultdict(int), excludes=None, negate=False):
        sentence = ''; weights = []; formula = ''
        if excludes is None:
            excludes = []
        for i, prod in enumerate(self._prod[symbol]):
            skip = False
            for token in excludes:
                if token in prod:
                    weights.append(0.01)
                    skip = True
                    break
            if skip:
                continue
            elif prod in pcount:
                weights.append(cfactor ** (pcount[prod]))
            else:
                weights.append(1.0)
        # sample for a production
        rand_prod = self._prod[symbol][self._weighted_choice(weights)]
        pcount[rand_prod] += 1
        if rand_prod in self._prod.keys():
            sentence, formula = self.gen_random(rand_prod, cfactor=cfactor, pcount=pcount, excludes=excludes, negate=negate)
        else:
            sentence, formula = self.gen_single_prod(rand_prod, cfactor=cfactor, pcount=pcount, excludes=excludes, negate=negate)
        if 'UOp' not in rand_prod and 'BinOp' not in rand_prod and \
                symbol in ['Safety', 'Guarantee', 'Recurrence', 'Persistence']:
            formula = '( ' + CLASS_LTL_PREFIX[symbol] + formula + ' )'
        # backtracking: clear the modification to pcount
        pcount[rand_prod] -= 1
        exclude = None

        sentence = sentence.replace("to do not", "to not")
        sentence = sentence.replace("to you should not", "to not")
        sentence = sentence.replace("you do not be", "you not be")
        return sentence, formula

    def gen_sentence(self, n=1):
        return [self.gen_random('S') for _ in range(n)]


if __name__ == '__main__':
    grammar = SentenceGrammar()
    for sentence, formula in grammar.gen_sentence(n=10):
        print('Sentence:', sentence)
        print('  LTL:', formula)
        alphabets = ['boundary', 'C_boundary', 'tree', 'C_tree', 'house', 'C_house', 'flag', 'C_flag', 'orange', 'C_orange','apple', 'C_apple', 'pear', 'C_pear']
        Automaton(formula, alphabets, add_flexible_state=False) 

