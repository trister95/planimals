import ast

def get_lst_from_csv(value):
  """
  Would be smart to do some more checking,
  since I expected that the first value declaratiod
  wouldn't be needed (but it turns out it is).
  """
  value = ast.literal_eval(value)
  value_str = value.decode('utf-8')
  return ast.literal_eval(value_str)

