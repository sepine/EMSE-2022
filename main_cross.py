#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time : 2021/4/1 15:39
@Author : Kunsong Zhao
"""

from trainer_cross import Trainer


params = {'out_path': './results_cross',
          'ext': '.xlsx',
          'flag': 'flag',
          'is_validate': False,
          'steps': 10,
          }


if __name__ == '__main__':

    trainer = Trainer(params)
    trainer.test_all()

