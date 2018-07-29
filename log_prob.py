from preprocess import *
from lm_train import *
from math import log


def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
    Compute the LOG probability of a sentence, given a language model and whether or not to
    apply add-delta smoothing

    INPUTS:
    sentence :	(string) The PROCESSED sentence whose probability we wish to compute
    LM :		(dictionary) The LM structure (not the filename)
    smoothing : (boolean) True for add-delta smoothing, False for no smoothing
    delta : 	(float) smoothing parameter where 0<delta<=1
    vocabSize :	(int) the number of words in the vocabulary

    OUTPUT:
    log_prob :	(float) log probability of sentence
    """
    # TODO: Implement by student.
    log_proba = 0
    i = 0
    sentence_split = sentence.split()
    if not smoothing:
        # return the MLE of the sentence, Obtain likelihoods by dividing bi-gram counts by uni-gram counts.
        while i < len(sentence_split)-1:
            if sentence_split[i] in LM["uni"]:
                denominator = LM["uni"][sentence_split[i]]
                if sentence_split[i + 1] in LM["bi"][sentence_split[i]]:
                    numerator = LM["bi"][sentence_split[i]][sentence_split[i+1]]
                    log_proba += log(numerator / denominator, 2)
                else:
                    # numerator = 0
                    log_proba += float('-inf')
            else:
                # denominator = 0
                log_proba += float('-inf')
            i += 1
        print("Not Smoothing Situation")
    else:
        # when smoothing is True, return a delta-smoothed estimate of the sentence.
        # the argument delta and vocabSize must also be specified (0 < delta <= 1)
        # [ c(w_t, w_{t+1}) + delta ] / [ c(w_t) + delta * v ]
        while i < len(sentence_split)-1:
            if sentence_split[i] in LM["uni"]:
                denominator = LM["uni"][sentence_split[i]]
                if sentence_split[i + 1] in LM["bi"][sentence_split[i]]:
                    numerator = LM["bi"][sentence_split[i]][sentence_split[i+1]]
                    log_proba += log((numerator + delta) / (denominator + delta * vocabSize), 2)
                else:
                    # numerator = 0
                    log_proba += log(delta / (denominator + delta * vocabSize), 2)
            else:
                # denominator = 0
                log_proba += float('-inf')
            i += 1
        print("Smoothing Situation")

    print(log_proba)

    return log_proba
#
#
# if __name__ == "__main__":
#     print(None)
