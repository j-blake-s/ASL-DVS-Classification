## Imports ##
import torch
import os
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR

from args import parse_args
from data import get_all, load_data, get_files
from augment import aslAugment
from training import train, test
from utils import format, import_file, setup_dir

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

## Args ##
args = parse_args()
args.device = "cpu"

## Data ##
blake, james, peyton = get_all(args.data_path, args.dataset)
train_data, test_data = load_data(train=None, test=peyton, augment=None, verbose=args.verbose)

# train_loader = torch.utils.data.DataLoader(
#   dataset=train_data,
#   batch_size=args.batch_size,
#   shuffle=True,
#   drop_last=False,
#   pin_memory=True,
# )


test_loader = torch.utils.data.DataLoader(
  dataset=test_data,
  batch_size=args.batch_size,
  shuffle=True,
  drop_last=False,
  pin_memory=True,
)



def cnf_matrix(model_name, combined_classes=False):

  ## Model ##
  model_dir = os.path.join("runs",model_name)
  model_path = os.path.join(model_dir,"model.py")
  weight_path = os.path.join(model_dir, "weights/best.pt")
  
  if args.verbose: print("Loading Model: ", model_name)
  model, _, _, pred = import_file(model_path).load_model(args)
  model.load_state_dict(torch.load(weight_path, weights_only=True))




  ## Confusion Matrix ## 
  model.eval()
  y_true = []
  y_pred = []
  labels = list(range(args.classes))
  with torch.no_grad():
    # for i, (x, y) in enumerate(train_loader):
    #   x, y = x.to(args.device), y.to(args.device)
    #   x = x.to(torch.float32)
    #   y = y.to(torch.int8)

    #   out = pred(model(x))
    #   y_true = y_true + y.tolist()
    #   y_pred = y_pred + out.tolist()

    for i, (x, y) in enumerate(test_loader):
      x, y = x.to(args.device), y.to(args.device)
      x = x.to(torch.float32)
      y = y.to(torch.int8)

      out = pred(model(x))
      y_true = y_true + y.tolist()
      y_pred = y_pred + out.tolist()



  if combined_classes is False:
    labels = ["Tuesday", "Bathroom", "Name", "Weight", "Brown", "Beer", "Favorite", "Colors", "Hamburger", "Marriage"]
    def foo(x): return x
  else:
    labels = ["Tuesday-Bathroom", "Name-Weight", "Brown-Beer", "Favorite-Colors", "Hamburger-Marriage"]
    def foo(x): return [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5][x]
  # cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)
  ConfusionMatrixDisplay.from_predictions(list(map(foo,y_true)), list(map(foo,y_pred)), display_labels=labels, xticks_rotation=45.0)
  plt.savefig(f"figures/{model_name}.png", bbox_inches='tight')
  if args.verbose: print(f"Modle {model_name} saved...")




models = [
  # "Cnn_0_0",
  # "Cnn_1_0",
  "Cnn_2_0",
  "Cnn_3_0",
  # "Cnn_1_0",
  # "SpatialCnn_Final"
  # "SpikeCnn_0_0",
  # "SpikeCnn_1_0",
  # "SpikeCnn_2_0",
  # "SpikeCnn_3_0",
  # "SpikeCnn_4_0",
  # "SpikeCnn_5_0",
  # "SpikeCnn_5_1",
  # "SpikeCnn_5_2",
  # "SpikeCnn_5_3",
]

for name in models:
  cnf_matrix(name)