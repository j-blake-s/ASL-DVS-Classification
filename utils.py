
import importlib

def format(num):
  whole, decimal = str(num).split(".")
  num = (" " * (3-len(whole)))+num
  num = num+(" " * (2-len(decimal)))
  return num

def import_file(path): return importlib.import_module(".".join(path[:-3].split("/")))

