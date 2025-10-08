## Imports ##
import torch
from args import parse_args
from data import get_all, load_data
from training import train, test
from utils import format, import_file

  
## Args ##
args = parse_args()
print("="*50)
print(f"[Model] \t{args.model[:-3].split('/')[-1].upper()}")
print(f"[Dataset]\t{args.dataset.upper()}")
print(f"[Device]\t{args.device}")
print(f"[Epochs]\t{args.epochs}")
print(f"[Batch Size]\t{args.batch_size}")
print("="*50)

## Model ##
if args.verbose: print("Loading Model...")
model, opt, err, pred = import_file(args.model).load_model(args)

## Data ##
blake, james, peyton = get_all(args.data_path, args.dataset)
train_data, test_data = load_data(train=blake[:5] + james[:5], test=peyton[:5], verbose=args.verbose)

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


## Training ##
for epoch in range(args.epochs):
  print(" "*50,end="\r")
  # print(f'Epoch [{epoch+1}/{args.epochs}]')
  train_acc = train(model, train_loader, test_loader, opt, err, pred, args)
  test_acc = test(model, test_loader, pred, args)
  print(f'\rEpoch [{epoch+1}/{args.epochs}] Train: {format(round(train_acc, 2))}  Val: {format(round(test_acc,2))} ')