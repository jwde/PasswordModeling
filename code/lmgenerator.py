from math import log, exp, isnan
import random
from decimal import *

start_token = "<S>"
end_token = "</S>"

def Preprocess(corpus):
    return [[start_token] + [token for token in pwd] + [end_token] for pwd in corpus]

class BigramLM:
    def __init__(self, training_corpus):
        self.bigram_counts = {}
        self.unigram_counts = {}
        self.Train(training_corpus)

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
    def StringLogProbability(self, string):
        preprocessed = Preprocess([string])[0]
        logprob = 0
        for i in xrange(1, len(preprocessed)):
            unigram = preprocessed[i - 1]
            bigram = preprocessed[i]
            if unigram in self.unigram_counts and unigram in self.bigram_counts and bigram in self.bigram_counts[unigram]:
                logprob += log(self.bigram_counts[unigram][bigram]) - log(self.unigram_counts[unigram])
            else:
                logprob = float('-inf')
        return logprob

    def ExpectedGuesses(self, string):
        logprob = self.StringLogProbability(string)
        try:
            expectation = Decimal(-logprob).exp()
            return expectation if not isnan(expectation) else float('inf')
        except:
            return float('inf')

    def Generator(self):
        while True:
            yield self.GenerateSample()

    def SimplePrunedGenerator(self):
        tries = set()
        while True:
            pwd = self.GenerateSample()
            if not pwd in tries:
                tries.update([pwd])
                yield pwd

