def check_for_entities(sentence_lst, entity_lst, start_numbering):
    """
    This functions tags entities names ngrams up
    to a lenght of 3 parts. Check if there aren't
    longer names in the reference_lst.
    """
    sentence_lst = [word.lower() for word in sentence_lst]
    tag_lst = []
    i = 0
    while i < len(sentence_lst):
        word = sentence_lst[i]
        if i < len(sentence_lst) - 2:
            phrase = word + " " + sentence_lst[i + 1] + " " + sentence_lst[i + 2]
            if phrase in entity_lst:
                tag_lst.extend(
                    [start_numbering, start_numbering + 1, start_numbering + 1]
                )
                i += 3
                continue
        if i < len(sentence_lst) - 1:
            phrase = word + " " + sentence_lst[i + 1]
            if phrase in entity_lst:
                tag_lst.extend([start_numbering, start_numbering + 1])
                i += 2
                continue
        if word in entity_lst:
            tag_lst.append(start_numbering)
        else:
            tag_lst.append(0)
        i += 1
    return tag_lst
