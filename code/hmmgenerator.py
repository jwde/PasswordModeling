from math import log, exp, log1p, isnan
import random
from memoize import memoize
import nltk
from decimal import *

class HMMLM:
    def __init__(self, training_corpus, num_states):
        sequences = [[(c, "") for c in pwd] for pwd in training_corpus]
        symbols = list(set([c for pwd in training_corpus for c in pwd]))
        states = range(num_states)
        trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states, symbols=symbols)
        self.hmm = trainer.train_unsupervised(sequences)

    def Sample(self, range_start, range_end):
        pwd = self.hmm.random_sample(random.Random(), random.randint(range_start, range_end))
        pwd = "".join([e[0] for e in pwd])
        return pwd

    def StringProbability(self, pwd):
        return self.hmm.log_probability([(c, None) for c in pwd])

    def ExpectedGuesses(self, pwd):
        logprob = self.StringProbability(pwd)
        try:
            expectation = Decimal(-logprob).exp()
            return expectation if not isnan(expectation) else float('inf')
        except:
            return float('inf')

    def Generator(self):
        while True:
            pwd = self.hmm.random_sample(random.Random(), random.randint(4, 18))
            pwd = "".join([e[0] for e in pwd])
            yield pwd


        """
def BigramHMMGenerator(training_corpus, num_states):
    while True:
        pwd = hmm.random_sample(random.Random(), random.randint(4, 18))
        pwd = "".join([e[0] for e in pwd])
        yield pwd
    """

"""
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
    split_count = count / 2
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
        self.transition_probability = {state1: {state2: prob for (state2, prob) in zip(self.states, RandomLogProbs(state_count))} for state1 in self.states}
        self.end_probability = {state: prob for (state, prob) in zip(self.states, RandomLogProbs(state_count))}
        self.emission_probability = {state: {symbol: prob for (symbol, prob) in zip(vocabulary, RandomLogProbs(len(vocabulary)))} for state in self.states}

    @memoize
    def ForwardMatrix(self, pwd):
        bp = [{state: None for state in self.states} for c in pwd]

        # initialization
        bp[0] = {state: self.start_probability[state] + self.emission_probability[state][pwd[0]] for state in self.states}

        # recursion
        for i in xrange(1, len(pwd)):
            bp[i] = {state: SumLogProbs(map(lambda p: bp[i - 1][p] + self.transition_probability[p][state] + self.emission_probability[state][pwd[i]], bp[i - 1])) for state in self.states}

        return bp


    @memoize
    def BackwardMatrix(self, pwd):
        bp = [{state: None for state in self.states} for c in pwd]

        # initialization
        bp[len(pwd) - 1] = {state: self.end_probability[state] for state in self.states}

        # recursion
        for i in reversed(xrange(0, len(pwd) - 1)):
            bp[i] = {state: SumLogProbs(map(lambda n: bp[i + 1][n] + self.transition_probability[state][n] + self.emission_probability[n][pwd[i + 1]], bp[i + 1])) for state in self.states}

        return bp


    @memoize
    def ForwardProbability(self, step, state, pwd):
        matrix = self.ForwardMatrix(pwd)
        if state == wildcard_token:
            return SumLogProbs(matrix[step].values())
        return matrix[step][state]


    @memoize
    def BackwardProbability(self, step, state, pwd):
        matrix = self.BackwardMatrix(pwd)
        return matrix[step][state]

        
    @memoize
    def TimeStateProbability(self, step, state, pwd):
        return self.ForwardProbability(step, state, pwd) + \
               self.BackwardProbability(step, state, pwd) - \
               self.ForwardProbability(len(pwd) - 1, wildcard_token, pwd)

    @memoize
    def StateTransitionProbability(self, step, state1, state2, pwd):
        return self.ForwardProbability(step, state1, pwd) + \
               self.BackwardProbability(step + 1, state2, pwd) + \
               self.transition_probability[state1][state2] + \
               self.emission_probability[state2][pwd[step + 1]] - \
               self.ForwardProbability(len(pwd) - 1, wildcard_token, pwd)

    def ForwardBackward(self, corpus):
        # for now assume convergence in constant number of iterations
        for i in xrange(10):
            print "Starting forward backward pass %d" % i
            print "Expectation"
            # expectation 
            tsps = {state: {e: float("-inf") for e in self.o_vocabulary} for state in self.states}
            for pwd in corpus:
                for step in xrange(len(pwd)):
                    for state in self.states:
                        tsps[state][pwd[step]] = SumLogProbs([self.TimeStateProbability(step, state, pwd), tsps[state][pwd[step]]])

            stps = [[{state1: {state2: self.StateTransitionProbability(step, state1, state2, pwd) for state2 in self.states} for state1 in self.states} for step in xrange(len(pwd) - 1)] for pwd in corpus]

            # maximization
            print "Maximization"
            # transitions
            for state1 in self.states:
                for state2 in self.states:
                    self.transition_probability[state1][state2] = \
                        SumLogProbs([SumLogProbs([step[state1][state2] for step in stp]) for stp in stps]) - \
                        SumLogProbs([SumLogProbs([SumLogProbs(step[state1].values()) for step in stp]) for stp in stps])

            # emissions
            for state in self.states:
                for e in self.o_vocabulary:
                    self.emission_probability[state][e] = tsps[state][e] - \
                                                          SumLogProbs(tsps[state].values())
                    
        
            # reset memos
            self.ForwardMatrix.reset()
            self.BackwardMatrix.reset()
            self.ForwardProbability.reset()
            self.BackwardProbability.reset()
            self.TimeStateProbability.reset()
            self.StateTransitionProbability.reset()

    def GenerateSample(self):
        return "test"


def BigramHMMGenerator(training_corpus, num_states):
    vocabulary = set()
    for pwd in training_corpus:
        vocabulary.update(pwd)
    hmm = BigramHMM(vocabulary, num_states)
    hmm.ForwardBackward(training_corpus)
    while True:
        yield hmm.GenerateSample()
        """
