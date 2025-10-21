## Imports ##
import torch
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR

from args import parse_args
from data import get_all, load_data, get_files
from augment import aslAugment
from training import train, test
from utils import format, import_file, setup_dir

## Args ##
args = parse_args()
args.device = "cpu"


## Model ##
if args.verbose: print("Loading Model...")
model, opt, err, pred = import_file(args.model).load_model(args)
model.load_state_dict(torch.load(args.checkpoint, weights_only=True))

## Data ##
blake, james, peyton = get_all(args.data_path, args.dataset)
train_data, test_data = load_data(train=blake+james, test=peyton, augment=None, verbose=args.verbose)

train_loader = torch.utils.data.DataLoader(
  dataset=train_data,
  batch_size=args.batch_size,
  shuffle=True,
  drop_last=False,
  pin_memory=True,
)


test_loader = torch.utils.data.DataLoader(
  dataset=test_data,
  batch_size=args.batch_size,
  shuffle=True,
  drop_last=False,
  pin_memory=True,
)


## Confusion Matrix ## 
model.eval()
y_true = []
y_pred = []
labels = list(range(args.classes))
with torch.no_grad():
  for i, (x, y) in enumerate(train_loader):
    x, y = x.to(args.device), y.to(args.device)
    x = x.to(torch.float32)
    y = y.to(torch.int8)

    out = pred(model(x))
    y_true = y_true + y.tolist()
    y_pred = y_pred + out.tolist()

  for i, (x, y) in enumerate(test_loader):
    x, y = x.to(args.device), y.to(args.device)
    x = x.to(torch.float32)
    y = y.to(torch.int8)

    out = pred(model(x))
    y_true = y_true + y.tolist()
    y_pred = y_pred + out.tolist()

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def foo(x): return [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5][x]
# cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)
ConfusionMatrixDisplay.from_predictions(list(map(foo,y_true)), list(map(foo,y_pred)))
plt.savefig("reduced_classes.png")