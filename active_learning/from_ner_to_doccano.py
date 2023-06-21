def hf_output_for_doccano(sentences, classifier):
  json_lst = []
  for sentence in sentences:
    hf_output = classifier(sentence)
    labels = []
    for sentence_part in hf_output:
      if sentence_part["entity_group"] == "animals":
        label = [sentence_part["start"], sentence_part["end"], "ANIMAL"]
        labels.append(label)
    data = {"text":sentence, "labels":labels}
  return json_lst
