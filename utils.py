import os
import importlib
import shutil

def format(num):
  whole, decimal = str(num).split(".")
  ret = (" " * (3-len(whole)))+str(num)
  ret = ret+("0" * (2-len(decimal)))
  return ret

def import_file(path): return importlib.import_module(".".join(path[:-3].split("/")))

def setup_dir(args):

  path = f"runs/{args.name}"
  
  # Check / remove existing directory
  if os.path.exists(path):
    if args.name == "default":
      shutil.rmtree(path) # Remove default directory
    else:
      print(f"./{path} already exists!")
      print(f"Quiting...")
      quit()

  # Make directories
  os.makedirs(path, exist_ok=False)
  os.makedirs(f'{path}/weights', exist_ok=False)
  
  # Output log
  pw = PrintWriter(f'{path}/log.txt')

  # Save model class definition
  os.system(f'cp {args.model} {path}/model.py')

  # Where to save weights
  weight_path = f'{path}/weights/best.pt'

  return pw, weight_path

class PrintWriter():
  def __init__(self, file): 
    self.fn = file
    with open(self.fn, 'w') as f: pass
  def print(self, s, end="\n"):
    print(s, end=end)
    with open(self.fn, 'a') as f:
      f.write(f'{s}{end}')

