## Imports ##
import torch
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR

from args import parse_args
from data import get_all, load_data
from augment import aslAugment
from training import train, test
from utils import format, import_file, setup_dir

torch.autograd.set_detect_anomaly(True)

## Args ##
args = parse_args()


## Model ##
if args.verbose: print("Loading Model...")
model, opt, err, pred = import_file(args.model).load_model(args)
if args.checkpoint: model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
scheduler = CosineAnnealingLR(opt, args.epochs, 1e-5)


## Data ##
blake, james, peyton = get_all(args.data_path, args.dataset)
if args.no_augment is False: aug = aslAugment
else: aug = None
train_data, test_data = load_data(train=blake+james, test=peyton, augment=None, combine_classes=args.combine_classes, verbose=args.verbose)

if args.no_augment:
  augment = None
else:
  augment = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=20),
  ])

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


## Setup Output Directory ##
if args.verbose: print("Setup Output Directory...")
pw, weight_path = setup_dir(args)

## Print Args ##
pw.print("="*50)
pw.print(f"[Model] \t{args.model[:-3].split('/')[-1].upper()}")
pw.print(f"[Params] \t{params:,}")
pw.print(f"[Save Dir]\truns/{args.name}")
if args.checkpoint: pw.print(f"[Checkpoint] \t{args.checkpoint}")
pw.print(f"[Dataset]\t{args.dataset.upper()}")
pw.print(f"[Device]\t{args.device}")
pw.print(f"[Epochs]\t{args.epochs}")
pw.print(f"[Batch Size]\t{args.batch_size}")
pw.print("="*50)



## Training ##
best = 0
for epoch in range(args.epochs):
  print(" "*50,end="\r")
  train_acc = train(model, train_loader, augment, opt, err, pred, args)
  test_acc = test(model, test_loader, pred, args)
  if test_acc > best:
    best = test_acc
    torch.save(model.state_dict(), weight_path)
  scheduler.step()
  pw.print(f'\r Epoch [{epoch+1}/{args.epochs}] \tTrain: {format(round(100*train_acc, 2))}%  Val: {format(round(100*test_acc,2))}% ')