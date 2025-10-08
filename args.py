import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Args')

    # Data
    parser.add_argument('--data_path', default="/data/DATASETS/pseudoDvs", type=str, help='path to data directory')
    parser.add_argument('--dataset', default="dvs", type=str, help='Training dataset. One of (rgb, dvs)')

    # Training
    parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')

    args = parser.parse_args()
    return args

