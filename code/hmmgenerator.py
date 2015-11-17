from math import log, exp
import random

start_token = "<S>"
end_token = "</S>"

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
        self.emission_probability = {state: {symbol: prob for (symbol, prob) in zip(vocabulary, RandomLogProbs(len(vocabulary)))} for state in self.states}

#def ForwardBackward():
        

            """
class BigramHMM:
    def __init__(self):
        self.transitions = {}
        self.vocabulary = set()

    def Train(self, training_corpus):
        training_set = Preprocess(training_corpus)
        unigram_counts = {}
        bigram_counts = {}
        for pwd in training_set:
            for i in xrange(len(pwd) - 1):
                token = pwd[i]
                next_token = pwd[i + 1]
                if not token in unigram_counts:
                    unigram_counts[token] = 0
                unigram_counts[token] += 1
                if not (token, next_token) in bigram_counts:
                    bigram_counts[(token, next_token)] = 0
                bigram_counts[(token, next_token)] += 1
        for (token, next_token) in bigram_counts:
            self.transitions[(token, next_token)] = log(bigram_counts[(token, next_token)]) - \
                                                    log(unigram_counts[token])

    for pwd in training_set:
        for i in xrange(1, len(pwd) - 1):
            self.vocabulary.update([pwd[i]])

    # there are no observations, so we take a desired observation length and emit
    # state transitions as the observation
    def ViterbiGenerator(self, length):
        matrix = [{token: (token, None, None) for token in self.dictionary} for x in xrange(length)]
        # start token to first observation
        matrix[0] = {token: (token, self.transitions[(start_token, token)], None) \
                     for token in matrix[0] if (start_token, token) in self.transitions}

        # recursion step
        for i in xrange(1, length):
            for token in matrix[i]:
                current_max = float("-inf")
                current_bp = matrix[i - 1].itervalues().next()
                for prev_token in matrix[i - 1]:


def hmmGenerator(training_corpus):

    """
