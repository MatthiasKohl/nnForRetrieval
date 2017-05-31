# -*- encoding: utf-8 -*-

import types
import torch


def readMeanStd(fname):
    with open(fname) as f:
        mean = map(float, f.readline().split(' '))
        std = map(float, f.readline().split(' '))
    return mean, std


def fun_str(f):
    if f.__class__ in (types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType):
        return f.__name__
    else:
        return f.__class__.__name__


def trans_str(trans):
    return ','.join(fun_str(t) for t in trans.transforms)


def move_device(obj, device):
    if device >= 0:
        return obj.cuda()
    else:
        return obj.cpu()


def tensor_t(t, device, *sizes):
    return move_device(t(*sizes), device)


def tensor(device, *sizes):
    return tensor_t(torch.Tensor, device, *sizes)
