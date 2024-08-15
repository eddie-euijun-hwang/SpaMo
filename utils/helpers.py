import os
import av
import json
import torch
import glob
import importlib
import numpy as np
import math


def derangement(lst):
    while True:
        shuffled = lst[:]
        random.shuffle(shuffled)
        if all(original != shuffled[i] for i, original in enumerate(lst)):
            return shuffled


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def instantiate_from_config(config):
    """
    Instantiates an object based on a configuration.

    Args:
        config (dict): Configuration dictionary with 'target' and 'params'.

    Returns:
        object: An instantiated object based on the configuration.
    """
    if 'target' not in config:
        raise KeyError('Expected key "target" to instantiate.')
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """
    Get an object from a string reference.

    Args:
        string (str): The string reference to the object.
        reload (bool): If True, reload the module before getting the object.

    Returns:
        object: The object referenced by the string.
    """
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def create_mask(seq_lengths: list, device="cpu"):
    """
    Creates a mask tensor based on sequence lengths.

    Args:
        seq_lengths (list): A list of sequence lengths.
        device (str): The device to create the mask on.

    Returns:
        torch.Tensor: A mask tensor.
    """
    max_len = max(seq_lengths)
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device)[:, None]
    return mask.to(torch.bool)

