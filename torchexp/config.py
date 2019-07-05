import os.path
import sys
import random
from ast import literal_eval
from ruamel.yaml import YAML
import numpy as np
import torch as th
import gin


def check_gin_special(s):
    return s.startswith('@') or s.startswith('%')


def _transform(value):
    if isinstance(value, str) and check_gin_special(value):
        return value
    return repr(value)


def _read_yaml_macros(yaml_path):
    yaml = YAML()
    with open(yaml_path, 'r') as f:
        data = yaml.load(f)
    gin.parse_config([f'{k} = {_transform(v)}' for k, v in data.items()])


@gin.configurable
def manual_seed(seed=None):
    if seed is None:
        seed = th.initial_seed() & ((1 << 63) - 1)
        gin.bind_parameter('%seed', seed)
        # execute the function again to record the modification of gin config
        manual_seed()
        return
    random.seed(seed)
    np.random.seed((seed >> 32, seed % (1 << 32)))
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def parse_args(args=None):
    gin.parse_config('torchexp.config.manual_seed.seed = %seed')
    gin.bind_parameter('%seed', None)

    if args is None:
        args = sys.argv[1:]

    for arg in args:
        try:
            key, value = arg.split('=', maxsplit=1)
        except ValueError:
            raise ValueError(f'The argument `{arg}` is not accepted!'
                             ' All argument should be the form name=value,'
                             ' --yaml=config.yaml or --gin=config.gin')
        if key == '--yaml':
            _read_yaml_macros(value)
        elif key == '--gin':
            gin.parse_config_file(value)
        else:
            if not check_gin_special(value):
                try:
                    value = literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
                value = repr(value)
            gin.parse_config(f'{key} = {value}')

    manual_seed()


def dump(root, name):
    gin_path = os.path.join(root, f'{name}.gin')
    if not os.path.isfile(gin_path):
        with open(gin_path, 'w') as f:
            f.write(gin.operative_config_str())
    arg_path = os.path.join(root, f'{name}.args')
    if not os.path.isfile(arg_path):
        with open(arg_path, 'w') as f:
            print(' '.join(sys.argv[1:]), file=f)
