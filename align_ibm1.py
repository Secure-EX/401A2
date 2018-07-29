from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os
# TODO: COMMENT THIS
# import time


def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
    Implements the training of IBM-1 word alignment algoirthm.
    We assume that we are implemented P(foreign|english)

    INPUTS:
    train_dir : 	(string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider
    max_iter : 		(int) the maximum number of iterations of the EM algorithm -> 1000 as slide provides
    fn_AM : 		(string) the location to save the alignment model

    OUTPUT:
    AM :			(dictionary) alignment model structure

    The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word']
    is the computed expectation that the foreign_word is produced by english_word.

            LM['house']['maison'] = 0.5
    """
    # AM['eng word']['fre word'] holds the probability (not log probability)
    # of the word eng word aligning to fre word.
    # In this sense, AM is essentially the t distribution from class

    # Read training data
    raw_e_AM, raw_f_AM = read_hansard(train_dir, num_sentences)
    
    # Initialize AM uniformly
    AM = initialize(raw_e_AM, raw_f_AM)
    print("AM model done")

    # Iterate between E and M steps
    for i in range(max_iter):
        em_step(AM, raw_e_AM, raw_f_AM)
    print("EM iter " + str(max_iter) + " times done")

    # print(AM)

    # Save Model
    with open(fn_AM + '.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM
    
# ------------ Support functions --------------


def read_hansard(train_dir, num_sentences):
    """
    Read up to num_sentences from train_dir.

    INPUTS:
    train_dir : 	(string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider


    Make sure to preprocess!
    Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

    Make sure to read the files in an aligned manner.
    """
    # TODO
    # the blue cat <-> le chat bleu
    # the red dog <-> le chein rouge
    # try to build a dict-list style storage:
    # raw_e_AM = {the: [le, chat, bleu, chein, rouge],
    #             blue: [le, chat, bleu],
    #             cat: [le, chat, bleu],
    #             red: [le, chein, rouge],
    #             dog: [le, chein, rouge]}
    raw_e_AM = {}
    raw_f_AM = {}
    if os.path.exists(train_dir):
        print("Correct path...")
        # trains on all of the data les in data dir that end in either 'e' for English or 'f' for French
        for subdir, dirs, files in os.walk(train_dir):
            for file in files:
                sent_num = 0
                # language is only 'e' files needs parallel process
                if os.path.basename(file)[-1] == "e":
                    file1 = os.path.basename(file)
                    file2 = os.path.basename(file)[:-1] + "f"
                    # open files in parallel way
                    if os.path.exists(train_dir + file2):
                        with open(train_dir + file1) as f1, open(train_dir + file2) as f2:
                            # preprocess every lines
                            for x, y in zip(f1, f2):
                                if sent_num < num_sentences:
                                    line1 = preprocess(x, "e").split()
                                    line2 = preprocess(y, "f").split()
                                    # block of raw_e_AM[sent_num][list of e]
                                    raw_e_AM[sent_num] = line1
                                    # block of raw_f_AM[sent_num][list of f]
                                    raw_f_AM[sent_num] = line2
                                    sent_num += 1
    else:
        print("Path " + train_dir + " does not exist ...")

    # print(raw_e_AM)
    # print(raw_f_AM)
    return raw_e_AM, raw_f_AM


def initialize(eng, fre):
    """
    Initialize alignment model uniformly.
    Only set non-zero probabilities where word pairs appear in corresponding sentences.
    """
    # TODO
    # eng and fre are two DICT type stuff that input in this function, have to use this function
    # try to build P(F|E) dictionary AM[eng][fre] = a probability that is 1 / numbers of fre
    # 1. initialize uniformly across rows, make a table of P(F|E) for all possible pairs F and E
    init_dict = {"SENTSTART": {"SENTSTART": 1}, "SENTEND": {"SENTEND": 1}}
    sub_init_dict = {}
    for sent_num in eng:
        line1 = eng[sent_num]
        line2 = fre[sent_num]
        # block of P(F|E)
        for word_e in line1:
            # if not in then create an empty
            # raw_AM = {the: []}
            if word_e not in sub_init_dict:
                if word_e != "SENTSTART" and word_e != "SENTEND":
                    sub_init_dict[word_e] = []
                    for word_f in line2:
                        if word_e in sub_init_dict:
                            if word_f not in sub_init_dict[word_e]:
                                if word_f != "SENTSTART" and word_f != "SENTEND":
                                    # raw_e_AM = {the: []}
                                    sub_init_dict[word_e].append(word_f)
                                else:
                                    continue
                            else:
                                continue
            else:
                for word_f in line2:
                    if word_f not in sub_init_dict[word_e]:
                        if word_f != "SENTSTART" and word_f != "SENTEND":
                            # aw_AM = {the: []}
                            sub_init_dict[word_e].append(word_f)
                        else:
                            continue
                    else:
                        continue

    # Initialize them uniformly -> generate all the P(F|E)
    for key_e in sub_init_dict:
        init_dict[key_e] = {}
        for key_f in sub_init_dict[key_e]:
            init_dict[key_e][key_f] = 1 / len(sub_init_dict[key_e])

    # print(init_dict)
    return init_dict


def e_f_count(lst):
    """
    INPUT:
    lst         (list) list of english or france words in sentence

    OUTPUT:
    d :			(dictionary) a dictionary of words appears count, remove "SENTSTART" and "SENTEND"
    """
    d = {}
    for w in lst:
        if w != "SENTSTART" and w != "SENTEND":
            if w in d:
                d[w] += 1
            else:
                d[w] = 1
    return d


def em_step(t, eng, fre):
    """
    One step in the EM algorithm.
    Follows the pseudo-code given in the tutorial slides.
    """
    # TODO
    # P(F|a, E) = \PI P(F_j|E_a_j)
    # Count(f_j, e_i) = P(a|E, F) = P(F|a, E) / \sum P(F|a, E)
    # dict -> TCount = \sum P(a|E, F)
    # dict -> total_e = \sum TCount = \sum \sum P(a|E, F)
    # new P(F|E) = TCount / total_e

    # t is AM = initialize(eng, fre)
    # set t_count(f, e) to 0 for all f, e
    t_count = {}
    # set total(e) to 0 for all e
    total_e = {}
    for key_e in eng:
        # iterate words
        # for each sentence pair (F, E) in training corpus
        # build a e_dictionary for easily calculate e count
        e_dict = e_f_count(eng[key_e])
        # build a f_dictionary for easily calculate f count
        f_dict = e_f_count(fre[key_e])
        # for each unique word f in F
        for word_f in list(f_dict.keys()):
            denom_c = 0
            # for each unique word e in E
            for word_e in list(e_dict.keys()):
                # denom_c += P(f|e) * F.count(f)
                denom_c += t[word_e][word_f] * f_dict[word_f]
            # for each unique word e in E
            for word_e in list(e_dict.keys()):
                # tcount(f, e) += P(f|e) * F.count(f) * E.count(e) / denom_c
                temp_t_count = t[word_e][word_f] * f_dict[word_f] * e_dict[word_e] / denom_c
                if word_f in t_count:
                    if word_e in t_count[word_f]:
                        t_count[word_f][word_e] += temp_t_count
                    else:
                        t_count[word_f][word_e] = temp_t_count
                else:
                    t_count[word_f] = {}
                    t_count[word_f][word_e] = temp_t_count
                # total(e) += P(f|e) * F.count(f) * E.count(e) / denom_c
                temp_total = t[word_e][word_f] * f_dict[word_f] * e_dict[word_e] / denom_c
                if word_e in total_e:
                    total_e[word_e] += temp_total
                else:
                    total_e[word_e] = temp_total
    # for each e in domain(total(:))
    for e in list(total_e.keys()):
        # for each f in domain(t_count(:,e))
        for f in list(t_count.keys()):
            if e in t_count[f]:
                # P(f|e) = t_count(f, e) / total(e)
                t[e][f] = t_count[f][e] / total_e[e]


# if __name__ == "__main__":
#     start_time = time.time()
#
#     # /u/cs401/A2_SMT/data/Hansard/Training/
#     # /u/cs401/A2_SMT/data/Toy/
#     # /h/u13/c4/00/xieruiyu/401A2/part_4_test/
#     # read_hansard(r"/u/cs401/A2_SMT/data/Toy/", 1000)
#     # read_hansard(r"/h/u13/c4/00/xieruiyu/401A2/part_4_test/", 1000)
#     # read_hansard(r"/h/u13/c4/00/xieruiyu/401A2/part_4_test_1/", 1000)
#
#     # max iter in range 5-25 as TA said on piazza
#     # align_ibm1(r"/h/u13/c4/00/xieruiyu/401A2/part_4_test/", 1000, 25,
#     #            r"/h/u13/c4/00/xieruiyu/401A2/temp_4")
#     # align_ibm1(r"/h/u13/c4/00/xieruiyu/401A2/part_4_test_1/", 1000, 25,
#     #            r"/h/u13/c4/00/xieruiyu/401A2/temp_4")
#     align_ibm1(r"/Users/ruiyuanxie/Desktop/UofT/CSC401 2018Winter/A2/data/Training/", 1000, 25,
#                r"/Users/ruiyuanxie/Desktop/UofT/CSC401 2018Winter/A2/Task5")
#
#     print("--- %s seconds to finish pre-processing ---" % (time.time() - start_time))
