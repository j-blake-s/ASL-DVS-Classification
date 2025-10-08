import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Args')

    # General
    parser.add_argument('--device', default="cuda", type=str, help='GPU to use')

    # Model
    parser.add_argument('--model', default=None, type=str, help='path to model python file')

    # Data
    parser.add_argument('--data_path', default="/data/DATASETS/pseudoDvs", type=str, help='path to data directory')
    parser.add_argument('--dataset', default="dvs", type=str, help='Training dataset. One of (rgb, dvs)')
    parser.add_argument('--classes', default=10, type=int, help='Number of Classes')

    # Training
    parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')
    parser.add_argument('-b','--batch_size', default=16, type=int, help='Size of training batches')

    # Output
    parser.add_argument('-v', '--verbose', action="store_true", help='Verbosity')

    args = parser.parse_args()
    args.channels = 2 if args.dataset == "dvs" else 3
    return args

