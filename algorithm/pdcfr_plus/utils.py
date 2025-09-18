import datetime
import importlib
import inspect
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Type

import numpy as np
import torch


def run_method(
    method: Callable,
    possible_args: Dict[str, Any],
    **kwargs,
):
    possible_args_copy = deepcopy(possible_args)
    for k, v in kwargs.items():
        possible_args_copy[k] = v
    args = inspect.getfullargspec(method).args
    params = {k: v for k, v in possible_args_copy.items() if k in args}
    result = method(**params)
    return result

