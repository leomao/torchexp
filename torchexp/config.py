import os.path
import sys
import random
from ruamel.yaml import YAML
import numpy as np
import torch as th
import gin


def _read_yaml_macros(yaml_path):
    yaml = YAML()
    with open(yaml_path, 'r') as f:
        data = yaml.load(f)
    gin.parse_config([f'{k} = {repr(v)}' for k, v in data.items()])


def _transform(value):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startwith('@'):
        return value

    return repr(value)


@gin.configurable
def manual_seed(seed=None):
    if seed is None:
        seed = th.initial_seed() % (1 << 63)
        gin.bind_parameter('%seed', seed)
        # excute the function again to trace the modification of config
        manual_seed()
        return
    random.seed(seed)
    np.random.seed((seed >> 32, seed % (1 << 32)))
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def parse_args(args=None):
    gin.bind_parameter('torchexp.config.manual_seed.seed', '%seed')
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
            value = _transform(value)
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
