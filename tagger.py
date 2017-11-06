#!/usr/bin/env python3
"""
ANLP A5: HMM for Part-of-Speech Tagging

Usage: 
  python tagger.py baseline
  python tagger.py hmm

(Adapted from Richard Johansson)
"""
from math import log, isfinite
from collections import Counter

import sys, os, time, platform, nltk

# utility functions to read the corpus

def read_tagged_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence

def read_tagged_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_tagged_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_tagged_sentence(f)
    return sentences


# utility function for color-coding in terminal
# https://gist.github.com/ssbarnea/1316877
def accepts_colors(handle=sys.stdout):
    if (hasattr(handle, "isatty") and handle.isatty()) or \
        ('TERM' in os.environ and os.environ['TERM']=='ANSI'):
        if platform.system()=='Windows' and not ('TERM' in os.environ and os.environ['TERM']=='ANSI'):
            return False #handle.write("Windows console, no ANSI support.\n")
        else:
            return True
    else:
        return False


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

ALPHA = .1
allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because missing Counter entries default to 0, not log(0)
transitionDists = {}
emissionDists = {}

def learn(tagged_sentences):
    """
    Record the overall tag counts (allTagCounts) and counts for each word (perWordTagCounts) for baseline tagger.
    (These should not have pseudocounts and should only apply to observed words/tags, not START, END, or UNK.)
    
    Learn the parameters of an HMM with add-ALPHA smoothing (ALPHA = 0.1):
     - Store counts + pseudocounts of observed transitions (transitionCounts) and emissions (emissionCounts) for bigram HMM tagger. 
     - Also store a pseudocount for UNK for each distribution.
     - Normalize the distributions and store (natural) log probabilities in transitionDists and emissionDists.
    """

    # store training data counts in allTagCounts, perWordTagCounts, transitionCounts, emissionCounts
    for sentence in tagged_sentences:
        prev=sentence[0]
        for word in sentence[1:]:
#            print(word)
            allTagCounts[word[1]]+=1
            
            if word[0] not in perWordTagCounts.keys():
                perWordTagCounts[word[0]]=Counter()
            perWordTagCounts[word[0]][word[1]]+=1
            
            if prev[1] not in transitionCounts.keys():
                transitionCounts[prev[1]]=Counter()
            transitionCounts[prev[1]][word[1]]+=1
            
            if word[1] not in emissionCounts.keys():
                emissionCounts[word[1]]=Counter()
            emissionCounts[word[1]][word[0]]+=1
    
    # add pseudocounts in transitionCounts and emissionCounts, including for UNK
    for sentence in tagged_sentences:
        s=(START, 'START')
        e=(END, 'END')
        wordS=sentence[0]
        wordE=sentence[-1]
        
        if s[1] not in transitionCounts.keys():
            transitionCounts[s[1]]=Counter()
        transitionCounts[s[1]][wordS[1]]+=1
        if s[1] not in emissionCounts.keys():
            emissionCounts[s[1]]=Counter()
        emissionCounts[s[1]][s[0]]+=1
        if e[1] not in transitionCounts.keys():
            transitionCounts[e[1]]=Counter()
        transitionCounts[e[1]][wordE[1]]+=1
        if e[1] not in emissionCounts.keys():
            emissionCounts[e[1]]=Counter()
        emissionCounts[e[1]][e[0]]+=1
        
    transitionCounts['UNK']=Counter()
    emissionCounts['UNK']=Counter()
    
    #add-alpha smoothing
    for tag in transitionCounts:
        for tag1 in transitionCounts[tag]:
            transitionCounts[tag][tag1]+=.1
        transitionCounts[tag]['UNK']+=.1
    for tag in emissionCounts:
        for word in emissionCounts[tag]:
            emissionCounts[tag][word]+=.1
        emissionCounts[tag]['UNK']+=.1
        

    # normalize counts and store log probability distributions in transitionDists and emissionDists
    for tag in transitionCounts:
        total=sum(transitionCounts[tag].values())
        transitionDists[tag]={}
        for tag1 in transitionCounts[tag]:
            transitionDists[tag][tag1]=log(transitionCounts[tag][tag1]/total)
    for tag in emissionCounts:
        total=sum(emissionCounts[tag].values())
        emissionDists[tag]={}
        for word in emissionCounts[tag]:
            emissionDists[tag][word]=log(emissionCounts[tag][word]/total)

def baseline_tag_sentence(sentence):
    """
    Tag the sentence with a most-frequent-tag baseline: 
    For each word, if it has been seen in the training data, 
    choose the tag it was seen most often with; 
    otherwise, choose the overall most frequent tag in the training data.
    Hint: Use the most_common() method of the Counter class.
    
    Do NOT modify the contents of 'sentence'.
    Return a list of (word, predicted_tag) pairs.
    """
    predictions=[]
    for word in sentence:
        word=word[0]
        if word in perWordTagCounts.keys():
            pred = max(k for k, v in perWordTagCounts[word].items())
            predictions.append((word, pred))
        else:
            predictions.append((word, max(k for k, v in allTagCounts.items())))
#    print(predictions)
    return predictions

def hmm_tag_sentence(sentence):
    """
    Tag the sentence with the bigram HMM using the Viterbi algorithm.
    Do NOT modify the contents of 'sentence'.
    Return a list of (word, predicted_tag) pairs.
    """
    # fill in the Viterbi chart
    ...
    
    # then retrace your steps from the best way to end the sentence, following backpointers
    ...
    
    # finally return the list of tagged words
    
    return ...



def viterbi(sentence):
    """
    Creates the Viterbi chart, column by column. 
    Each column is a list of tuples representing cells.
    Each cell ("item") holds: the tag being scored at the current position; 
    a reference to the corresponding best item from the previous position; 
    and a log probability. 
    This function returns the END item, from which it is possible to 
    trace back to the beginning of the sentence.
    """
    # make a dummy item with a START tag, no predecessor, and log probability 0
    # current list = [ the dummy item ]
    current = ...
    
    # for each word in the sentence:
    #    previous list = current list
    #    current list = []        
    #    determine the possible tags for this word
    #  
    #    for each tag of the possible tags:
    #         add the highest-scoring item with this tag to the current list
    
    ...

    # end the sequence with a dummy: the highest-scoring item with the tag END
    return ...
    
def find_best_item(word, tag, possible_predecessors):    
    # determine the emission probability: 
    #  the probability that this tag will emit this word
    
    ...
    
    # find the predecessor that gives the highest total log probability,
    #  where the total log probability is the sum of
    #    1) the log probability of the emission,
    #    2) the log probability of the transition from the tag of the 
    #       predecessor to the current tag,
    #    3) the total log probability of the predecessor
    
    ...
    
    # return a new item (tag, best predecessor, best total log probability)
    return ...

def retrace(end_item, sentence_length):
    # tags = []
    # item = predecessor of end_item
    # while the tag of the item isn't START:
    #     add the tag of item to tags
    #     item = predecessor of item
    # reverse the list of tags and return it
    ...
    return ...

def joint_prob(sentence):
    """Compute the joint probability of the given words and tags under the HMM model."""
    p = 0   # joint log prob. of words and tags
    ...
    assert isfinite(p) and p<0  # Should be negative
    return p

def count_correct(gold_sentence, pred_sentence):
    """Given a gold-tagged sentence and the same sentence with predicted tags,
    return the number of tokens that were tagged correctly overall, 
    the number of OOV tokens tagged correctly, 
    and the total number of OOV tokens."""
    assert len(gold_sentence)==len(pred_sentence)
    ...
    return correct, correctOOV, OOV



TRAIN_DATA = 'en-ud-train.upos.tsv'
TEST_DATA = 'en-ud-test.upos.tsv'

train_sentences = read_tagged_corpus(TRAIN_DATA)
#print(train_sentences)


# train the bigram HMM tagger & baseline tagger in one fell swoop
trainingStart = time.time()
learn(train_sentences)
trainingStop = time.time()
trainingTime = trainingStop - trainingStart


# decide which tagger to evaluate
if len(sys.argv)<=1:
    assert False,"Specify which tagger to evaluate: 'baseline' or 'hmm'"
if sys.argv[1]=='baseline':
    tagger = baseline_tag_sentence
elif sys.argv[1]=='hmm':
    tagger = hmm_tag_sentence
else:
    assert False,'Invalid command line argument'



if accepts_colors():
    class bcolors:  # terminal colors
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
else:
    class bcolors:
        HEADER = ''
        OKBLUE = ''
        OKGREEN = ''
        WARNING = ''
        FAIL = ''
        ENDC = ''
        BOLD = ''
        UNDERLINE = ''


def render_gold_tag(x):
    (w,gold),(w,pred) = x
    return w + '/' + (bcolors.WARNING + gold + bcolors.ENDC if gold!=pred else gold)
    
def render_pred_tag(x):
    (w,gold),(w,pred) = x
    return w + '/' + (bcolors.FAIL + pred + bcolors.ENDC if gold!=pred else pred)




test_sentences = read_tagged_corpus(TEST_DATA)
#print(test_sentences)

nTokens = nCorrect = nOOV = nCorrectOOV = nPerfectSents = nPGoldGreater = nPPredGreater = 0

taggingTime = 0

for sent in test_sentences:
    taggerStart = time.time()
    pred_tagging = tagger(sent)
    taggerStop = time.time()
    taggingTime += taggerStop - taggerStart
    nCorrectThisSent, nCorrectOOVThisSent, nOOVThisSent = count_correct(sent, pred_tagging)
    
    acc = nCorrectThisSent/len(sent)
    
    pHMMGold = joint_prob(sent)
    pHMMPred = joint_prob(pred_tagging)
    print(pHMMGold, ' '.join(map(render_gold_tag, zip(sent,pred_tagging))))
    print(pHMMPred, ' '.join(map(render_pred_tag, zip(sent,pred_tagging))), '{:.0%}'.format(acc))
    
    if pHMMGold > pHMMPred:
        nPGoldGreater += 1
        #assert False
    elif pHMMGold < pHMMPred:
        nPPredGreater += 1
    
    nCorrect += nCorrectThisSent
    nCorrectOOV += nCorrectOOVThisSent
    nOOV += nOOVThisSent
    nTokens += len(sent)
    if pred_tagging==sent:
        nPerfectSents += 1

print('TAGGING ACCURACY BY TOKEN: {}/{} = {:.1%}   OOV TOKENS: {}/{} = {:.1%}   PERFECT SENTENCES: {}/{} = {:.1%}   #P_HMM(GOLD)>P_HMM(PRED): {}   #P_HMM(GOLD)<P_HMM(PRED): {}'.format(nCorrect, nTokens, nCorrect/nTokens, 
            nCorrectOOV, nOOV, nCorrectOOV/nOOV,
            nPerfectSents, len(test_sentences), nPerfectSents/len(test_sentences), 
            nPGoldGreater, nPPredGreater))
print('RUNTIME: TRAINING = {:.2}s, TAGGING = {:.2}s'.format(trainingTime, taggingTime))


