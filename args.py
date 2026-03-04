import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Args')

    # General
    parser.add_argument('--device', default="cuda", type=str, help='GPU to use')

    # Model
    parser.add_argument('--model', default="models/acnn.py", type=str, help='path to model python file')
    parser.add_argument('-c','--checkpoint', default=None, type=str, help='path to model checkpoint')

    # Data
    parser.add_argument('--data_path', default="/data/DATASETS/pseudoDvs/prep", type=str, help='path to data directory')
    parser.add_argument('--dataset', default="dvs", type=str, help='Training dataset. One of (rgb, dvs)')
    parser.add_argument('--classes', default=10, type=int, help='Number of Classes')
    parser.add_argument('-t', '--timesteps', default=45, type=int, help='number of timesteps')
    parser.add_argument('--no_augment', default=False, action="store_true", help='Does not augment the data')
    parser.add_argument('--combine-classes', action="store_true", help='Combine Similar Class Pairs')

    # Training
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('-b','--batch_size', default=8, type=int, help='Size of training batches')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate')
    parser.add_argument('--interval', default=3, type=int, help='size of timestep accumulation')

    # Output
    parser.add_argument('-v', '--verbose', default=True, action="store_true", help='Verbosity')
    parser.add_argument('-n', '--name', default="default", type=str, help='Name of trial run')

    args = parser.parse_args()
    args.channels = 2 if args.dataset == "dvs" else 3
    return args

