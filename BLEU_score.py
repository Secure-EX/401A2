import math


def BLEU_score(candidate, references, n):
    """
    Compute the LOG probability of a sentence, given a language model and whether or not to
    apply add-delta smoothing

    INPUTS:
    sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
    references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
    n :			(int) one of 1,2,3. N-Gram level.


    OUTPUT:
    bleu_score :	(float) The BLEU score
    """
    # TODO: Implement by student.
    if n == 1:
        # remove SENTSTART and SENTENDs
        candidate = candidate.split()
        candidate = candidate.remove("SENTSTART")
        candidate = candidate.remove("SENTEND")
        N = len(candidate)
        r1 = len(references[0]) - 2
        r2 = len(references[1]) - 2
        if N != 0:
            # build a ref words list that all appear, include duplicate
            mod_ref = []
            for ref_sent in references:
                for word in ref_sent.split():
                    if word != "SENTSTART" and word != "SENTEND":
                        mod_ref.append(word)
            # evaluate the candidate list and find the C
            i = 0
            exist_candi = []
            while i < len(candidate):
                if candidate[i] in mod_ref:
                    mod_ref = mod_ref.remove(candidate[i])
                    exist_candi.append(candidate[i])
            C = len(exist_candi)
            p1 = C / N
        else:
            p1 = 0
        # here N is same as c_i
        BP = 0
        if abs(r1 - N) < abs(r2 - N):
            if N != 0:
                brevity = r1 / N
                if brevity < 1:
                    BP = 1
                else:
                    BP = math.exp(1 - brevity)
            else:
                BP = 0
        elif abs(r1 - N) > abs(r2 - N):
            if N != 0:
                brevity = r1 / N
                if brevity < 1:
                    BP = 1
                else:
                    BP = math.exp(1 - brevity)
            else:
                BP = 0
        bleu_score = BP * p1
    if n == 2:
        # count unigram precision first
        # remove SENTSTART and SENTENDs
        candidate = candidate.split()
        candidate = candidate.remove("SENTSTART")
        candidate = candidate.remove("SENTEND")
        N = len(candidate)
        if N != 0:
            # build a ref words list that all appear, include duplicate
            mod_ref = []
            for ref_sent in references:
                for word in ref_sent.split():
                    if word != "SENTSTART" and word != "SENTEND":
                        mod_ref.append(word)
            # evaluate the candidate list and find the C
            i = 0
            exist_candi = []
            while i < len(candidate):
                if candidate[i] in mod_ref:
                    mod_ref = mod_ref.remove(candidate[i])
                    exist_candi.append(candidate[i])
            C = len(exist_candi)
            p1 = C / N
        else:
            p1 = 0
        # then count bigram precision in second place
        print("nah")
    if n == 3:
        # count unigram precision first
        # remove SENTSTART and SENTENDs
        candidate = candidate.split()
        candidate = candidate.remove("SENTSTART")
        candidate = candidate.remove("SENTEND")
        N = len(candidate)
        if N != 0:
            # build a ref words list that all appear, include duplicate
            mod_ref = []
            for ref_sent in references:
                for word in ref_sent.split():
                    if word != "SENTSTART" and word != "SENTEND":
                        mod_ref.append(word)
            # evaluate the candidate list and find the C
            i = 0
            exist_candi = []
            while i < len(candidate):
                if candidate[i] in mod_ref:
                    mod_ref = mod_ref.remove(candidate[i])
                    exist_candi.append(candidate[i])
            C = len(exist_candi)
            p1 = C / N
        else:
            p1 = 0
        # then count bigram precision in second place
        print("nah")

    return bleu_score
