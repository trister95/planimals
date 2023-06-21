from langdetect import detect
def check_for_organisms(sentence_lst, organism_lst):
    """
    This functions tags organisms names ngrams up
    to a lenght of 3 parts. Check if there aren't
    longer names in the reference_lst.
    """
    tag_lst = []
    i = 0
    while i < len(sentence_lst):
        word = sentence_lst[i]
        if i < len(sentence_lst) - 2:
            phrase = word + " " + sentence_lst[i+1] + " " + sentence_lst[i+2]
            if phrase in organism_lst:
                tag_lst.extend([1, 1, 1])
                i += 3
                continue
        if i < len(sentence_lst) - 1:
            phrase = word + " " + sentence_lst[i+1]
            if phrase in organism_lst:
                tag_lst.extend([1, 1])
                i += 2
                continue
        if word in organism_lst:
            tag_lst.append(1)
        else:
            tag_lst.append(0)
        i += 1
    return tag_lst
    
def find_language(sentence_lst):
    return detect((" ".join(sentence_lst)))

