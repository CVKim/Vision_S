from collections import abc
from importlib import import_module
from inspect import ismodule
from typing import Any, Callable, Optional, Type, Union


def is_seq_of(
    seq: Any, expected_type: Union[Type, tuple], seq_type: Type = None
) -> bool:
    """
    Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def get_object_from_string(obj_name: str):
    """
    Get object from name.

    Args:
        obj_name (str): The name of the object.

    Examples:
        >>> get_object_from_string('torch.optim.sgd.SGD')
        >>> torch.optim.sgd.SGD
    """
    parts = iter(obj_name.split("."))
    module_name = next(parts)
    # import module
    while True:
        try:
            module = import_module(module_name)
            part = next(parts)
            # mmcv.ops has nms.py and nms function at the same time. So the
            # function will have a higher priority
            obj = getattr(module, part, None)
            if obj is not None and not ismodule(obj):
                break
            module_name = f"{module_name}.{part}"
        except StopIteration:
            # if obj is a module
            return module
        except ImportError:
            return None

    # get class or attribute from module
    obj = module
    while True:
        try:
            obj = getattr(obj, part)
            part = next(parts)
        except StopIteration:
            return obj
        except AttributeError:
            return None
