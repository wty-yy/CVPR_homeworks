# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: test.py.py
@time: 2022/12/2 20:12
"""

import argparse as ap
import json
import configparser as cp

if __name__ == '__main__':
    config = cp.RawConfigParser()
    config.read(r'config.cfg')
    string = json.loads(config.get('hog', 'min_wdw_sz'))
    a = json.loads('[1,2,3]')
    print(a)