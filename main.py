














from data import get_all, load_data
from args import parse_args


args = parse_args()
print("="*50)
print(f"[Dataset]\t{args.dataset.upper()}")
print(f"[Epochs]\t{args.epochs}")
print("="*50)



blake, james, peyton = get_all(args.data_path, args.dataset)
train_data, test_data = load_data(train=blake[:5] + james[:5], test=peyton[:5], verbose=args.verbose)