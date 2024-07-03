#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test an audio classification model."""

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_net import test


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform multi-clip testing.
    launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
