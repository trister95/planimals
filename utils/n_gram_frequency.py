from collections import Counter
import ucto
configurationfile_ucto = "tokconfig-nld-historical"
tokenizer = ucto.Tokenizer(configurationfile_ucto)

def print_n_gram_frequence(tokenizer = tokenizer):
  gram_lst = []
  for n_gram in animal_lst:
    tokenizer.process(n_gram)
    length = sum(1 for _ in tokenizer)
    gram_lst.append(length)
  counts = Counter(gram_lst)
  for number, count in counts.items():
    print(f'The number {number} appears {count} times.')
  return
