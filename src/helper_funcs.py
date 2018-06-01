import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np

"""
Got from the Deep Image Prior Repo: https://github.com/DmitryUlyanov/deep-image-prior
"""




def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10, batch_size=1):
    """Returns a tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [batch_size, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input.data, noise_type)
        net_input.data *= var
        net_input = net_input.data
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        meshgrid = meshgrid[None,:]
        if batch_size > 1:
            meshgrid = np.repeat(meshgrid, batch_size, 0)
        net_input = torch.from_numpy(meshgrid).float()
    else:
        assert False

    return net_input


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(42)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False
