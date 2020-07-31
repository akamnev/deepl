import numpy as np
import torch


def _scatter_index(length, num):
    chunk_size = round(length / num)
    i = 0
    index = []
    while len(index) < num:
        chunk = []
        while len(chunk) < chunk_size and i < length:
            chunk.append(i)
            i += 1
        index.append(chunk)
    while i < length:
        index[-1].append(i)
        i += 1
    return index


def scatter(inputs, target_gpus, dim=0):
    """scatter is an adaptation of torch.nn.parallel.scatter_gather.scatter
    in the case when the input data of the model are lists
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            index = _scatter_index(len(obj), len(target_gpus))
            return [obj[i] for i in index]
        if isinstance(obj, np.ndarray):
            index = _scatter_index(len(obj), len(target_gpus))
            return [obj[i] for i in index]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            index = _scatter_index(len(obj), len(target_gpus))
            return [[obj[i] for i in idx] for idx in index]
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for _ in target_gpus]
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res


class DataParallel(torch.nn.DataParallel):
    """DataParallel is an adaptation of torch.nn.DataParallel
    in the case when the input data of the model are lists
    """
    def scatter(self, inputs, kwargs, device_ids, dim=0):
        inputs = scatter(inputs, device_ids, dim) if inputs else []
        kwargs = scatter(kwargs, device_ids, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs
