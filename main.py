#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time : 2021/4/1 15:39
@Author : Kunsong Zhao
"""

import argparse
import torch
from trainer import Trainer

import tensorflow

params = {'base_path': './datasets/',
          'out_path': './results',
          'ext': '.xlsx',
          'flag': 'flag',
          'is_validate': False,
          'split_ratio': 0.5,
          'steps': 50,
          }


if __name__ == '__main__':
    trainer = Trainer(params)
    trainer.test_all()

