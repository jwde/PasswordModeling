from math import log, exp
import random

start_token = "<S>"
end_token = "</S>"

def Preprocess(corpus):
    return [[start_token] + [token for token in pwd] + [end_token] for pwd in corpus]

class BigramLM:
    def __init__(self):
        self.bigram_counts = {}
        self.unigram_counts = {}

    def Train(self, training_corpus):
        training_set = Preprocess(training_corpus)
        for pwd in training_set:
            for i in xrange(len(pwd) - 1):
                token = pwd[i]
                next_token = pwd[i + 1]
                if not token in self.unigram_counts:
                    self.unigram_counts[token] = 0
                if not token in self.bigram_counts:
                    self.bigram_counts[token] = {}
                if not next_token in self.bigram_counts[token]:
                    self.bigram_counts[token][next_token] = 0
                self.unigram_counts[token] += 1
                self.bigram_counts[token][next_token] += 1

    def GenerateSample(self):
        sample = [start_token]
        while not sample[-1] == end_token:
            selector = random.uniform(0, self.unigram_counts[sample[-1]])
            sum_bc = 0
            for bigram in self.bigram_counts[sample[-1]]:
                sum_bc += self.bigram_counts[sample[-1]][bigram]
                if sum_bc > selector:
                    sample.append(bigram)
                    break
        return ''.join(sample[1:-1])

    # gets the (unsmoothed) probability of a string given the bigramlm
#    def StringLogProbability(self, string):


def BigramLMGenerator(training_corpus):
    lm = BigramLM()
    lm.Train(training_corpus)
    while True:
        yield lm.GenerateSample()

def SimplePrunedBigramLMGenerator(training_corpus):
    tries = set()
    gen = BigramLMGenerator(training_corpus)
    while True:
        pwd = gen.next()
        if not pwd in tries:
            tries.update([pwd])
            yield pwd
