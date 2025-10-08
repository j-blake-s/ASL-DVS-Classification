## Imports ##
from data import get_all, load_data
from args import parse_args
import torch

## Args ##
args = parse_args()
print("="*50)
print(f"[Dataset]\t{args.dataset.upper()}")
print(f"[Epochs]\t{args.epochs}")
print(f"[Batch Size]\t{args.batch_size}")
print(f"[Device]\t{args.device}")
print("="*50)


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

## Model ##
model = None


## Training ##
