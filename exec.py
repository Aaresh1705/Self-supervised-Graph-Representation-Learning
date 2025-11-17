#!/usr/bin/env python3

from pretrain_gmae import train_gmae
from pretrain_gae import train_gae
from supervised import train_paper_classifier 

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_arg("--model")
    subparsers = parser.add_subparsers()
    pretrain_parser = subparsers.add_parser("pretrain", func=pretrain)
    args = parser.parse_args()
    print(args)

if __name__ == "__main__":
    main()
