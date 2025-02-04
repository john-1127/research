"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.parsing import parse_fingerprint_args
from chemprop.train import create_fingerprints

if __name__ == '__main__':
    args = parse_fingerprint_args()
    if args.mpn_output_only == True:
        create_fingerprints(args)
