from preprocess import *
import pickle
import os
# TODO: need to comment this!
import time


def lm_train(data_dir, language, fn_LM):
    """
    This function reads data from data_dir, computes unigram and bigram counts,
    and writes the result to fn_LM

    INPUTS:

    data_dir	: (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language	: (string) either 'e' (English) or 'f' (French)
    fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT

    LM			: (dictionary) a specialized language model

    The file fn_LM must contain the data structured called "LM", which is a dictionary
    having two fields: 'uni' and 'bi', each of which holds sub-structures which
    incorporate unigram or bigram counts

    e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
          LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
    # Some reminders from piazza

    # *** You should be adding SENTSTART and SENTEND before and after the sentence (in preprocessing). ***
    # A B C
    # -> SENTSTART A B C SENTEND
    # White space is not counted as a token.

    # >>> LM = {}
    # >>> LM = {"uni": {"word": 5, "bird": 2}, "bi": {"word": {"bird": 2}, "bird":{"word": 0}}}
    # >>> LM["uni"]["word"]
    # 5
    # >>> LM["bi"]["word"]["bird"]
    # 2

    # e.g. ['SENTSTART', 'a', 'b', 'c', 'SENTEND']
    # LM = {"uni": {'SENTSTART': 1, "a": 1, "b": 1, "c": 1, 'SENTEND': 1},
    #       "bi": {'SENTSTART': {'a': 1}, 'a': {'b': 1}, 'b': {'c': 1}, 'c': {'SENTEND': 1}}}

    language_model = {"uni": {}, "bi": {}}

    # TODO: Implement Function
    if os.path.exists(data_dir):
        print("Correct path...")
        # trains on all of the data les in data dir that end in either 'e' for English or 'f' for French
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                file_name = os.path.basename(file)
                # print(file_name)
                # language is either 'e' (English) or 'f' (French)
                if file_name[-1] == language:
                    with open(data_dir + file_name, "r") as f:
                        for line in f:
                            i = 0
                            modified = preprocess(line, file_name[-1])
                            key = modified.split()
                            # print(key)
                            while i+1 <= len(key):
                                # use slicing to deal with uni-gram and bi-gram
                                # uni-gram block
                                if key[i] not in language_model["uni"]:
                                    language_model["uni"][key[i]] = 1
                                elif key[i] in language_model["uni"]:
                                    language_model["uni"][key[i]] += 1

                                # bi-gram block
                                if key[i] not in language_model["bi"] and i+1 < len(key):
                                    bi_gram_dict = {}
                                    bi_gram_dict[key[i+1]] = 1
                                    language_model["bi"][key[i]] = bi_gram_dict
                                elif key[i] in language_model["bi"]:
                                    if key[i+1] in language_model["bi"][key[i]] and i+1 < len(key):
                                        language_model["bi"][key[i]][key[i+1]] += 1
                                    else:
                                        language_model["bi"][key[i]][key[i + 1]] = 1
                                i += 1
        # print(language_model['uni'])
        # print(language_model['bi'])

        # Save Model
        with open(fn_LM + '.pickle', 'wb') as handle:
            pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # print(language_model)
        print(language + " language model done")
        return language_model

    else:
        print("Path " + data_dir + " does not exist ...")


# if __name__ == "__main__":
#     start_time = time.time()
#
#     # /u/cs401/A2_SMT/data/Hansard/Training/
#     # /u/cs401/A2_SMT/data/Toy
#     # lm_train(r"/u/cs401/A2_SMT/data/Toy/",
#     #          "e",
#     #          r"/h/u13/c4/00/xieruiyu/401A2/out_put/my_toy_e")
#     # lm_train(r"/u/cs401/A2_SMT/data/Toy/",
#     #          "f",
#     #          r"~/401A2/out_put/my_toy_f")
#
#     print("--- %s seconds to finish pre-processing ---" % (time.time()
#                                                            - start_time))
