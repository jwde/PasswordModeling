"""
    Test the performance of various password models
"""

import lmgenerator

# basic wordlist attack
def baselineGenerator(training_corpus):
    for pwd in training_corpus:
        yield pwd
    while True:
        yield ""

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
    bigramlmgen = lmgenerator.SimplePrunedBigramLMGenerator(training_corpus)
    bigramlm = testGenerator(bigramlmgen, test_corpus, tries)
    print "Bigram LM attack -- %d tries: %d." % (tries, bigramlm)
 

def main():
    print "################################################################"
    print "Training corpus: rockyou"
    print "Test corpus: gmail"
    print "################################################################"
    rockyou_nocount = open('corpora/rockyou_nocount', 'r')
    training_corpus = [pwd.rstrip() for pwd in rockyou_nocount]
    gmail_nocount = open('corpora/gmail_nocount', 'r')
    gmail_corpus = [pwd.rstrip() for pwd in gmail_nocount]
    test_corpus = gmail_corpus[:-5000]
    held_out_corpus = gmail_corpus[-5000:]
    testCorpora(training_corpus, test_corpus)


if __name__ == "__main__":
    main()
