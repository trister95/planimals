# should place this somewhere else

from collections import Counter
import ucto

configurationfile_ucto = "tokconfig-nld-historical"  # if I remove count_n_gram_frequencies, I can remove this
tokenizer = ucto.Tokenizer(configurationfile_ucto)


def print_n_gram_frequency(lst):
    """
    Prints the frequency of different lengths of n-grams in a list.
    """

    gram_lengths = []
    for n_gram in lst:
        tokenizer.process(n_gram)
        length = sum(1 for _ in tokenizer)
        gram_lengths.append(length)

    counts = Counter(gram_lengths)
    for number, count in counts.items():
        print(f"The number {number} appears {count} times.")
