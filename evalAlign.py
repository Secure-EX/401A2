from BLEU_score import *
from preprocess import *
from log_prob import *
from lm_train import *
from decode import *
from math import log
import pickle
import os
import re
# TODO: COMMENT THIS
import time


def evalAlign(train_dir, fn_AM, fn_LM, test_dir, max_iter, num_sentences, bleu_n):
    """
    Produce your own translations, obtain reference translations from Google and the Hansards,
    and use the latter to evaluate the former, with a BLEU score

    :param train_dir:
    :param fn_AM:
    :param fn_LM:
    :param test_dir:
    :param max_iter:
    :param num_sentences:
    :return:
    """
    bleu = 0
    # generate LM and AM
    LM = lm_train(train_dir, "e", fn_LM)
    print("LM generated")
    AM = align_ibm1(train_dir, num_sentences, max_iter, fn_AM)
    print("AM generated")
    # test block
    if os.path.exists(test_dir):
        print("Correct path...")
        # find all the test data, and find Task5.f only
        for subdir, dirs, files in os.walk(test_dir):
            for file in files:
                file_name = os.path.basename(file)
                # print(file_name)
                # test file name must be "Task5.f"
                if file_name == "Task5.f":
                    with open(test_dir + file_name, "r") as f:
                        for line in f:
                            french = preprocess(line, "f")
                            # this is the candidate
                            english = decode(french, LM, AM)
                            # print("ENG: " + english)


    else:
        print("Path " + test_dir + " does not exist ...")


if __name__ == "__main__":
    # /u/cs401/A2 SMT/data/Hansard/Testing/Task5.f
    # 1. /u/cs401/A2 SMT/data/Hansard/Testing/Task5.e
    # 2. /u/cs401/A2 SMT/data/Hansard/Testing/Task5.google.e
    # FOR CDF
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 1000, 1))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 1000, 2))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 1000, 3))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 10000, 1))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 10000, 2))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 10000, 3))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 15000, 1))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 15000, 2))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 15000, 3))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 30000, 1))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 30000, 2))
    # print(evalAlign(r"/u/cs401/A2_SMT/data/Hansard/Training/",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5AM",
    #                 r"/h/u13/c4/00/xieruiyu/401A2/Task5LM",
    #                 r"/u/cs401/A2 SMT/data/Hansard/Testing/", 25, 30000, 3))
    # FOR MY PC
    print(evalAlign(r"/Users/ruiyuanxie/Desktop/UofT/CSC401 2018Winter/A2/data/Training/",
                    r"/Users/ruiyuanxie/Desktop/UofT/CSC401 2018Winter/A2/Task5AM",
                    r"/Users/ruiyuanxie/Desktop/UofT/CSC401 2018Winter/A2/Task5LM",
                    r"/Users/ruiyuanxie/Desktop/UofT/CSC401 2018Winter/A2/data/Testing/",
                    25,
                    1000,
                    1))
