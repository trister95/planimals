from langdetect import detect
def find_language_in_ucto_tokenized_sentence(sentence_lst):
    return detect((" ".join(sentence_lst)))
