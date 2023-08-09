import ucto
import json



import spacy
from spacy.training import offsets_to_biluo_tags, biluo_to_iob
nlp = spacy.blank("nl")


def jsonl_to_ner_format(filepath, tag2id, text_col="text", label_col="label"):
  lst = []
  with open(filepath, 'r') as file:
    for line in file:
      data = json.loads(line)
      text, entities = data[text_col], data[label_col]
      doc = nlp(text)
      biluo_tags = offsets_to_biluo_tags(doc, entities)
      iob_tags = biluo_to_iob(biluo_tags)
      try:
        number_tags = [tag2id[iob] for iob in iob_tags]
        together = [[str(word) for word in doc],number_tags]
        lst.append(together)
      except:
        print(doc)
  return lst

#evrything below is probably not needed anymore






configurationfile_ucto = "tokconfig-nld-historical"
tokenizer = ucto.Tokenizer(configurationfile_ucto)

def number_of_tagged_words(target_word):
    """ 
    Please not that this function needs a tokenizer!
    """
    tokenizer.process(target_word)
    word_lst = [str(token) for token in tokenizer]
    return len(word_lst)

def tag_one_entity(splitted, target_word, landing_strip, tagged):
    number_of_t = number_of_tagged_words(target_word)
    for count, _ in enumerate(splitted):
        t_words = splitted[count:(count+number_of_t)]
        splitted_minus_target_words = [x for x in splitted if x not in t_words]
        rest = "".join(splitted_minus_target_words)  
        if "".join(t_words) == target_word.replace(" ","") and rest == landing_strip:
            for i in range(count, count+number_of_t):
                tagged[i]=1
            break
    return tagged

def add_splitted_to_doccano(json_obj, df):
    sentence = json_obj["text"]
    row = df[df["sentence"] == sentence].iloc[0]  # Get the corresponding row
    splitted_sentence = row["splitted"]
    json_obj["splitted"] = splitted_sentence
    return json_obj

def hf_tag_one_obj(obj, df):
    obj = add_splitted_to_doccano(obj, df)
    sentence, labels, splitted = obj["text"], obj["label"], obj["splitted"]
    tagged = [0]*len(splitted)

    for la in labels:
        start, end, _ = la
        target_word = sentence[start:end]
        landing_strip = (sentence[:start] + sentence[end:]).replace(" ", "")
        tagged = tag_one_entity(splitted, target_word, landing_strip, tagged)
    obj["hf_tags"] = tagged
    return obj   

def add_hf_tags(path_to_jsonl, df):
    json_lst = []
    with open(path_to_jsonl, "r") as file:
        for line in file:
            obj = json.loads(line)
            hf_tags_added = hf_tag_one_obj(obj, df)
            json_lst.append(hf_tags_added)
    return json_lst
