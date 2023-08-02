import json
from tqdm import tqdm
def to_doccano_format(sentences, classifier):
  json_lst = []
  for sentence in tqdm(sentences):
    hf_output = classifier(sentence)
    if bool(hf_output) == True:
      labels = [[d["start"], d["end"], d["entity_group"]] for d in hf_output]
      data ={"text":sentence, "labels":labels}
      json_lst.append(json.dumps(data))
  return json_lst
