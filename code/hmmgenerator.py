from math import log, exp, log1p, isnan
import random
import nltk
from decimal import *

class NLTKHMMLM:
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


start_token = "<S>"
end_token = "</S>"
wildcard_token = "<*>"

# reduce floating point imprecision in adding probabilities in log space
def SumLogProbs(lps):
    # ln(e^lp1 + e^lp2) == ln(e^lp2 (e^(lp1 - lp2) + 1)) = ln(e^(lp1 - lp2) + 1) + lp2
    def adderhelper(lp1, lp2):
        if lp1 == float('-inf') and lp2 == float('-inf'):
            return float('-inf')
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


class HMMLM:
    def __init__(self, training_corpus, state_count):
        vocabulary = set()
        for pwd in training_corpus:
            vocabulary.update([c for c in pwd])
        self.o_vocabulary = set(vocabulary)
        self.states = range(state_count)
        self.transition_probability = {state1: {state2: prob for (state2, prob) in zip(self.states + [end_token], RandomLogProbs(state_count + 1))} for state1 in self.states}
        self.transition_probability[start_token] = {state: prob for (state, prob) in zip(self.states, RandomLogProbs(state_count))}
        self.emission_probability = {state: {symbol: prob for (symbol, prob) in zip(vocabulary, RandomLogProbs(len(vocabulary)))} for state in self.states}
        self.emission_probability[start_token] = {start_token: 0}
        self.emission_probability[end_token] = {end_token: 0}
        self.ForwardBackward(training_corpus)


    def ForwardMatrix(self, pwd):
        bp = [{state: None for state in self.states} for c in pwd]

        # initialization
        bp.insert(0, {start_token: 0})
        bp.append({end_token: None})
        ppwd = [start_token] + [c for c in pwd] + [end_token]

        # recursion
        for i in xrange(1, len(pwd) + 2):
            bp[i] = {state: SumLogProbs(map(lambda p: bp[i - 1][p] + self.transition_probability[p][state] + self.emission_probability[state][ppwd[i]], bp[i - 1])) for state in bp[i]}

        return bp
        

    def BackwardMatrix(self, pwd):
        bp = [{state: None for state in self.states} for c in pwd]

        # initialization
        bp.append({end_token: 0})
        bp.insert(0, {start_token: None})
        ppwd = [start_token] + [c for c in pwd] + [end_token]

        # recursion
        for i in reversed(xrange(0, len(pwd) + 1)):
            bp[i] = {state: SumLogProbs(map(lambda n: bp[i + 1][n] + self.transition_probability[state][n] + self.emission_probability[n][ppwd[i + 1]], bp[i + 1])) for state in bp[i]}

        return bp

    def TransitionMatrix(self, alpha, beta, pwd):
        length = len(pwd) + 2
        ppwd = [start_token] + [c for c in pwd] + [end_token]
        m = [{i: {} for i in self.states + [start_token]} for t in xrange(length - 1)]
        for t in xrange(length - 1):
            for start in self.states + [start_token]:
                for end in self.states + [end_token]:
                    if start not in alpha[t] or end not in beta[t + 1]:
                        m[t][start][end] = float('-inf')
                    else:
                        m[t][start][end] = alpha[t][start] + self.transition_probability[start][end] + self.emission_probability[end][ppwd[t + 1]] + beta[t + 1][end] - alpha[length - 1][end_token]
        return m


    def TimeStateMatrix(self, alpha, beta, pwd):
        length = len(pwd) + 2
        m = [{} for t in xrange(length)]
        for t in xrange(length):
            for j in self.states + [start_token, end_token]:
                if j not in alpha[t] or j not in beta[t]:
                    m[t][j] = float('-inf')
                else:
                    m[t][j] = alpha[t][j] + beta[t][j] - alpha[length - 1][end_token]
        return m


    def ForwardBackward(self, corpus):
        max_length = max([len(pwd) for pwd in corpus]) + 2
        last_prob = None
        x = 0
        while True:
            print "iteration:", x
            log_prob = 0

            ksi_all = [{i: {j: float('-inf') for j in self.states + [end_token]} for i in self.states + [start_token]} for t in xrange(max_length - 1)]
            gamma_all = [{j: {} for j in self.states + [start_token, end_token]} for t in xrange(max_length)]

            for pwd in corpus:
                alpha = self.ForwardMatrix(pwd)
                beta = self.BackwardMatrix(pwd)
                ksi = self.TransitionMatrix(alpha, beta, pwd)
                gamma = self.TimeStateMatrix(alpha, beta, pwd)
                length = len(pwd) + 2
                log_prob += alpha[length - 1][end_token]
                for t in xrange(length - 1):
                    for start in self.states + [start_token]:
                        for end in self.states + [end_token]:
                            ksi_all[t][start][end] = SumLogProbs([ksi_all[t][start][end], ksi[t][start][end]])
                ppwd = [start_token] + [c for c in pwd] + [end_token]
                for t in xrange(length):
                    for j in self.states + [start_token, end_token]:
                        if ppwd[t] not in gamma_all[t][j]:
                            gamma_all[t][j][ppwd[t]] = float('-inf')
                        gamma_all[t][j][ppwd[t]] = SumLogProbs([gamma_all[t][j][ppwd[t]], gamma[t][j]])

            # re-estimate transition probabilities using ksi_all
            for start in self.states + [start_token]:
                transition_den = float('-inf')
                for end in self.states + [end_token]:
                    for t in xrange(max_length - 1):
                        transition_den = SumLogProbs([transition_den, ksi_all[t][start][end]])
                for end in self.states + [end_token]:
                    transition_num = float('-inf')
                    for t in xrange(length - 1):
                        transition_num = SumLogProbs([transition_num, ksi_all[t][start][end]])
                    self.transition_probability[start][end] = transition_num - transition_den

            # re-estimate emission probabilities using gamma_all
            for j in self.states + [start_token, end_token]:
                emission_den = float('-inf')
                for t in xrange(max_length):
                    emission_den = SumLogProbs([emission_den] + gamma_all[t][j].values())
                for e in self.emission_probability[j]:
                    emission_num = float('-inf')
                    for t in xrange(max_length):
                        if e in gamma_all[t][j]:
                            emission_num = SumLogProbs([emission_num, gamma_all[t][j][e]])
                    self.emission_probability[j][e] = emission_num - emission_den

            print "log prob:", log_prob
            if last_prob is not None and log_prob - last_prob < .0001:
                break
            last_prob = log_prob
            x += 1


    def GenerateSample(self):
        state = start_token
        sample = []
        while len(sample) < 18:
            transitions = [(next_state, Decimal(self.transition_probability[state][next_state]).exp()) for next_state in self.transition_probability[state]]
            selector = random.uniform(0, 1)
            sum_p = 0
            for (next_state, dec_prob) in transitions:
                sum_p += dec_prob
                if sum_p >= selector:
                    state = next_state
                    break
            if state == end_token:
                break
            emissions = [(e, Decimal(self.emission_probability[state][e]).exp()) for e in self.emission_probability[state]]
            selector = random.uniform(0, 1)
            sum_p = 0
            for (e, dec_prob) in emissions:
                sum_p += dec_prob
                if sum_p >= selector:
                    sample.append(e)
                    break
        return ''.join(sample)

    def Generator(self):
        while True:
            yield self.GenerateSample()



