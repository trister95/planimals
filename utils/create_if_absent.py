def create_if_absent(directory):
  """
  mia = make if absent
  Function creates a directory if it doesn't exist already.
  """
  import os
  if not os.path.exists(directory):
    os.makedirs(directory)
    print("Directory", directory, "created")
  else:
    print("Directory", directory, "already exists")
  return