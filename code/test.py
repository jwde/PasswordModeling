"""
    Test the performance of various password models
"""

import lmgenerator
import hmmgenerator

# basic wordlist attack
def baselineGenerator(training_corpus):
    for pwd in training_corpus:
        yield pwd
    vocabulary = set()
    for pwd in training_corpus:
        for c in pwd:
            vocabulary.update([c])
    vocabulary = list(vocabulary)
    def bruteForce(vocabulary, length):
        if length == 1:
            for c in vocabulary:
                yield c
        else:
            for c in vocabulary:
                for password_end in bruteForce(vocabulary, length - 1):
                    yield c + password_end
    length = 1
    while True:
        for pwd in bruteForce(vocabulary, length):
            yield pwd
        length += 1

def passwordStrength(password, training_corpus, bigramlm, hmm):
    expectations = []
    # baseline
    baseline_guess = 0
    if password in training_corpus:
        baseline_guess = training_corpus.index(password) + 1
    else:
        vocabulary = set()
        for pwd in training_corpus:
            for c in pwd:
                vocabulary.update([c])
        vocabulary = list(vocabulary)
        guessable = True
        for c in password:
            if c not in vocabulary:
                guessable = False
        if not guessable:
            baseline_guess = float('inf')
        else:
            baseline_guess = len(training_corpus)
            for pwd_len in xrange(1, len(password)):
                baseline_guess += pow(len(vocabulary), pwd_len)
            def helper(pwd):
                if len(pwd) == 1:
                    return vocabulary.index(pwd[0]) + 1
                return pow(len(vocabulary), len(pwd) - 1) * vocabulary.index(pwd[0]) + helper(pwd[1:])
            baseline_guess += helper(password)
    expectations.append(baseline_guess)

    if bigramlm:
        expectations.append(bigramlm.ExpectedGuesses(password))
    if hmm:
        expectations.append(hmm.ExpectedGuesses(password))
    return min(expectations)

# See how many things in test_corpus the generator can guess with some number of
# tries
def testGenerator(gen, test_corpus, tries):
    found = 0
    test_set = set(test_corpus)
    guesses = set()
    for i in xrange(tries):
        guess = gen.next()
        if not guess in guesses:
            guesses.update([guess])
            if guess in test_set:
                found += 1
    return found

def testCorpora(training_corpus, test_corpus):
    print "First 5 training passwords: ", training_corpus[:5]
    print "First 5 test passwords: ", test_corpus[:5]

    tries = 100000
    baseline = testGenerator(baselineGenerator(training_corpus), test_corpus, tries)
    print "Baseline wordlist attack -- %d tries: %d." % (tries, baseline)
    bigramlm = lmgenerator.BigramLM(training_corpus)
    bigramlmgen = bigramlm.Generator()
    bigramlmresults = testGenerator(bigramlmgen, test_corpus, tries)
    print "Bigram LM attack -- %d tries: %d." % (tries, bigramlmresults)
    hmmlm = hmmgenerator.NLTKHMMLM(training_corpus, 100)
    hmmgen = hmmlm.Generator()
    hmmresults = testGenerator(hmmgen, test_corpus, tries)
    print "HMM attack -- %d tries: %d." % (tries, hmmresults)
    print "Password strength test:"
    print "123456:", passwordStrength("123456", training_corpus, bigramlm, hmmlm)
    print "123456789123:", passwordStrength("123456789123", training_corpus, bigramlm, hmmlm)
    print "mingchow:", passwordStrength("mingchow", training_corpus, bigramlm, hmmlm)
    print "0a0I8DV:", passwordStrength("0a0I8DV", training_corpus, bigramlm, hmmlm)
    print "AtiK0nAOLP3y:", passwordStrength("AtiK0nAOLP3y", training_corpus, bigramlm, hmmlm)
    print "correcthorsebatterystaple:", passwordStrength("correcthorsebatterystaple", training_corpus, bigramlm, hmmlm)

 

def main():
    print "################################################################"
    print "Training corpus: rockyou"
    print "Test corpus: gmail"
    print "################################################################"
    rockyou_nocount = open('corpora/rockyou_nocount', 'r')
    training_corpus = [pwd.rstrip() for pwd in rockyou_nocount][:10]
    gmail_nocount = open('corpora/gmail_nocount', 'r')
    gmail_corpus = [pwd.rstrip() for pwd in gmail_nocount]
    test_corpus = gmail_corpus[:-5000]
    held_out_corpus = gmail_corpus[-5000:]
    testCorpora(training_corpus, test_corpus)


if __name__ == "__main__":
    main()
