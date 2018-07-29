import re


def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation

    INPUTS:
    in_sentence : (string) the original sentence to be processed
    language	: (string) either 'e' (English) or 'f' (French) Language of in_sentence

    OUTPUT:
    out_sentence: (string) the modified sentence
    """
    # pre-modified the raw sentences, strip start and the end spaces, convert to lower
    out_sentence = in_sentence.lower()
    out_sentence = out_sentence.strip()

    # abbrev -> (c-)(.*) or a.m. or p.m. or ref. or lib. or

    # a temporary storage for modified sentence
    mod_sent = out_sentence
    for steps in range(1, 7):
        # separate dashes between parentheses, ignore G-7 or C-17 etc.
        if steps == 1:
            mod_sent = re.sub(r"([a-z])(-)([a-z])", r"\1 \2 \3", mod_sent)
        # ''
        if steps == 2:
            mod_sent = re.sub(r"('')", r" \1 ", mod_sent)
        # ``
        if steps == 3:
            mod_sent = re.sub(r"(``)", r" \1 ", mod_sent)
        # multiple numbers, 57, 88,
        if steps == 4:
            mod_sent = re.sub(r"(^[a-z-][\d]+^[a-z-])", r" \1 ", mod_sent)
        # 1998-2000
        if steps == 5:
            mod_sent = re.sub(r"(\d)(-)(\d)", r" \1 \2 \3 ", mod_sent)
        # the most greedy, separate sentence-final punctuation, commas, colons and semicolons, parentheses
        if steps == 6:
            mod_sent = re.sub(r"([!\"#$%&()*+,./:;<=>?@\[\]^{|}~])", r" \1 ", mod_sent)

    if language == "e":
        # Clitics separate
        mod_sent = re.sub(r"(\w+)(n't)", r"\1 \2", mod_sent)
        mod_sent = re.sub(r"(\w+)('\w*)", r"\1 \2", mod_sent)
        # print("e")
    if language == "f":
        mod_sent = re.sub(r"(b|c|f|g|h|j|k|l|m|n|p|q|r|s|t|v|x|z|qu)(')(\w+)", r" \1\2 \3", mod_sent)
        mod_sent = re.sub(r"(\w+)(')(on|il)$", r"\1\2 \3", mod_sent)
        # print("f")

    # You should be adding SENTSTART and SENTEND before and after the sentence (in preprocessing).
    # A B C
    # -> SENTSTART A B C SENTEND
    # White space is not counted as a token.
    mod_sent = mod_sent.split()
    mod_sent.insert(0, "SENTSTART")
    mod_sent.append("SENTEND")
    out_sentence = " ".join(mod_sent)

    return out_sentence

if __name__ == "__main__":
    in_sentence = "I would like to quote some advice I read on February 10 of this year. "
    print(preprocess(in_sentence, "e"))
    in_sentence = "(Kamouraska-Riviere-du-Loup-Temiscouata-Les, Lib.):"
    print(preprocess(in_sentence, "f"))
    in_sentence = "puisqu'on, loraqu'il"
    print(preprocess(in_sentence, "f"))
