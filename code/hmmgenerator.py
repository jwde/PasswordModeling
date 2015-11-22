from math import log, exp, log1p
import random
from memoize import memoize

start_token = "<S>"
end_token = "</S>"
wildcard_token = "<*>"

# reduce floating point imprecision in adding probabilities in log space
def SumLogProbs(lps):
    # ln(e^lp1 + e^lp2) == ln(e^lp2 (e^(lp1 - lp2) + 1)) = ln(e^(lp1 - lp2) + 1) + lp2
    def adderhelper(lp1, lp2):
        return log1p(exp(lp1 - lp2)) + lp2 if lp2 > lp1 else log1p(exp(lp2 - lp1)) + lp1
    return reduce(adderhelper, lps)


def Preprocess(corpus):
    return [[start_token] + [token for token in pwd] + [end_token] for pwd in corpus]

# gets a count-length array of random probabilities summing to s
def RandomPartition(count, s):
    if count is 1:
        return [s]
    split_prob = (random.random() * .4 + .2) * s
    split_count = len(count) / 2
    return RandomPartition(split_count, split_prob) + \
           RandomPartition(count - split_count, s - split_prob)

# gets an array of log probabilities [p1, p2, ...] where e^p1 + e^p2 + ... = 1
def RandomLogProbs(count):
    total = 4000000000
    partition = RandomPartition(count, total)
    return [log(p) - log(total) for p in partition]


class BigramHMM:
    def __init__(self, vocabulary, state_count):
        self.o_vocabulary = set(vocabulary)
        self.states = range(state_count)
        self.start_probability = {state: prob for (state, prob) in zip(self.states, RandomLogProbs(state_count))}
        self.transition_probability = {state1: {state2: prob for (state2, prob) in (self.states, RandomLogProbs(state_count))} for state1 in self.states}
        self.end_probability = {state: prob for (state, prob) in zip(self.states, RandomLogProbs(state_count))}
        self.emission_probability = {state: {symbol: prob for (symbol, prob) in zip(vocabulary, RandomLogProbs(len(vocabulary)))} for state in self.states}

    @memoize
    def ForwardMatrix(pwd):
        bp = [{state: None for state in self.states} for c in pwd]

        # initialization
        bp[0] = {state: self.start_probability[state] + self.emission_probability[state][pwd[0]] for state in self.states}

        # recursion
        for i in xrange(1, len(pwd)):
            bp[i] = {state: SumLogProbs(map(lambda p: bp[i - 1][p] + self.transition_probability[p][state] + self.emission_probability[state][pwd[i]], bp[i - 1])) for state in self.states}

        return bp


    @memoize
    def BackwardMatrix(pwd):
        bp = [{state: None for state in self.states} for c in pwd]

        # initialization
        bp[len(pwd) - 1] = {state: self.end_probability[state] for state in self.states}

        # recursion
        for i in reversed(xrange(0, len(pwd) - 1)):
            bp[i] = {state: SumLogProbs(map(lambda n: bp[i + 1][n] + self.transition_probability[state][n] + self.emission_probability[n][pwd[i + 1]], bp[i + 1])) for state in self.states}

        return bp


    @memoize
    def ForwardProbability(step, state, pwd):
        matrix = self.ForwardMatrix(pwd)
        if state == wildcard_token:
            return SumLogProbs(matrix[step].values())
        return matrix[step][state]
        """
        if step is 0:
            return self.start_probability[end_state]

        bp = [{state: None for state in self.states} for c in pwd]

        # initialization
        bp[0] = {state: self.start_probability[state] + self.emission_probability[state][pwd[0]] for state in self.states}

        # recursion
        for i in xrange(1, step - 1):
            bp[i] = {state: sum(map(lambda p: bp[i - 1][p] + self.transition_probability[p][state] + self.emission_probability[state][pwd[i]], bp[i - 1])) for state in self.states}

        # termination
        if end_state == wildcard_token:
            return sum(map(lambda state: sum(map(lambda p: bp[step - 1][p] + self.transition_probability[p][state] + self.emission_probability[state][pwd[step]], bp[step - 1])), bp[step]))

        return sum(map(lambda p: bp[step - 1][p] + self.transition_probability[p][end_state] + self.emission_probability[end_state][pwd[step]], bp[step - 1]))
    """

    @memoize
    def BackwardProbability(step, state, pwd):
        matrix = self.BackwardMatrix(pwd)
        return matrix[step][state]
        """
        last_step = len(pwd) - 1
        if step == last_step:
            return self.end_probability[start_state]
        
        bp = [{state: None for state in self.states} for c in pwd]

        # initialization
        bp[last_step] = {state: self.end_probability[state] for state in self.states}

        # recursion
        for i in reversed(xrange(step + 1, last_step - 1)):
            bp[i] = {state: sum(map(lambda n: bp[i + 1][n] + self.transition_probability[state][n] + self.emission_probability[n][pwd[i + 1]], bp[i + 1])) for state in self.states}

        # termination
        return sum(map(lambda n: bp[step + 1][n] + self.transition_probability[start_state][n] + self.emission_probability[n][pwd[step + 1]], bp[step + 1]))
    """
        
    @memoize
    def TimeStateProbability(step, state, pwd):
        return self.ForwardProbability(step, state, pwd) + \
               self.BackwardProbability(step, state, pwd) - \
               self.ForwardProbability(len(pwd) - 1, wildcard_token, pwd)

    @memoize
    def StateTransitionProbability(step, state1, state2, pwd):
        return self.ForwardProbability(step, state1, pwd) + \
               self.BackwardProbability(step + 1, state2, pwd) + \
               self.transition_probability[state1][state2] + \
               self.emission_probability[state2][pwd[step + 1]] - \
               self.ForwardProbability(len(pwd) - 1, wildcard_token, pwd)

    def ForwardBackward():
        # for now assume convergence in constant number of iterations
        for i in xrange(10):
            # expectation 
            
            # maximization
        
            # reset memos
            self.ForwardMatrix.reset()
            self.BackwardMatrix.reset()
            self.ForwardProbability.reset()
            self.BackwardProbability.reset()
            self.TimeStateProbability.reset()
            self.StateTransitionProbability.reset()
