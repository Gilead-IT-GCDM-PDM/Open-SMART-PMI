import argparse

from train import train
from predict import predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose",
        help="make output more verbose",
        action="store_true"
    )
    parser.add_argument(
        "cmd", type=str,
        choices=["train", "predict"],
        help="train model or make prediction using saved model"
    )
    parser.add_argument(
        "-i", "--input", type=str,
        help="input directory for data used for training or predictions",
        default="."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str,
        help="output directory to save trained model or predictions",
        default="."
    )
    parser.add_argument(
        "-m", "--model", type=str,
        help="path to load model for making predictions",
        default='' # option to use default model
    )
    args = parser.parse_args()
    print('verbose: ', args.verbose)
    print('cmd: ', args.cmd)
    print('model: ', args.model)
    print('input: ', args.input)
    print('output: ', args.output_dir)

    if args.cmd == 'train':
        train(args.input, args.output_dir)
    else:
        predict(args.input, args.output_dir, args.model)
