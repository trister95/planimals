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
