#!/usr/bin/env python
# coding: utf-8

# Choy, C.; Gwak, J.; Savarese, S. 4D Spatio Temporal ConvNet: Minkowski Convolutional Neural Networks.
# Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.


import torch

from examples.minkunet import MinkUNet34C


__all__ = [
    'Coordinates',
    'CoordinatesColors',
]

STATE_DICT_PREFIX = 'STATE_DICT_EXPERIMENT_ITERATION_'
STATE_DICT_SUFIX = '.pt'
MODELS_BASE_URL = 'http://hosting.wu.ac.at/reklasat3d/models/weights'
model_urls_base = {
    'coordinates': f'{MODELS_BASE_URL}/coordinates',
    'coordinates_colors': f'{MODELS_BASE_URL}/coordinates_colors'
}

# Number of in in_channels for coordinates.
C_IN_CHANNELS = 3
# Number of in_channels for coordinates and colors.
CC_IN_CHANNELS = 6
# Dimensionality of space.
SPACE_DIM = 3
# Number of classes.
NUM_CLASSES = 6


def coordinates(progress=True):
    '''
    Returns the model MinkUNet34C from Choey et al. trained on coordinates
    (in_channels=3) with weights after 50 Epochs.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    '''
    EPOCH = 50

    model = MinkUNet34C(in_channels=C_IN_CHANNELS, out_channels=NUM_CLASSES, D=SPACE_DIM)

    url = f'{model_urls_base["coordinates"]}/{STATE_DICT_PREFIX}{EPOCH}{STATE_DICT_SUFIX}'

    state_dict = torch.hub.load_state_dict_from_url(url,
                                                    progress=progress)

    model.load_state_dict(state_dict)

    return model


def coordinates_epoch(epoch=50, progress=True):
    '''
    Returns the model MinkUNet34C from Choey et al. trained on coordinates
    (in_channels=3) with weights after <epoch> Epochs.

    Args:
        epoch (int): Specifies the class weights to be returned.
        progress (bool): If True, displays a progress bar of the download to stderr.
    '''
    model = MinkUNet34C(in_channels=C_IN_CHANNELS, out_channels=NUM_CLASSES, D=SPACE_DIM)

    url = f'{model_urls_base["coordinates"]}/{STATE_DICT_PREFIX}{epoch}{STATE_DICT_SUFIX}'

    state_dict = torch.hub.load_state_dict_from_url(url,
                                                    progress=progress)

    model.load_state_dict(state_dict)
    return model


def coordinates_colors(progress=True):
    '''
    Returns the model MinkUNet34C from Choey et al. trained on coordinates
    and colors (in_channels=6) with weights after 50 Epochs.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    '''
    EPOCH = 50

    model = MinkUNet34C(in_channels=CC_IN_CHANNELS, out_channels=NUM_CLASSES, D=SPACE_DIM)

    url = f'{model_urls_base["coordinates_colors"]}/{STATE_DICT_PREFIX}{EPOCH}{STATE_DICT_SUFIX}'

    state_dict = torch.hub.load_state_dict_from_url(url,
                                                    progress=progress)

    model.load_state_dict(state_dict)

    return model


def coordinates_colors_epoch(epoch=50, progress=True):
    '''
    Returns the model MinkUNet34C from Choey et al. trained on coordinates
    and colors (in_channels=6) with weights after <epoch> Epochs.

    Args:
        epoch (int): Specifies the class weights to be returned.
        progress (bool): If True, displays a progress bar of the download to stderr.
    '''
    model = MinkUNet34C(in_channels=CC_IN_CHANNELS, out_channels=NUM_CLASSES, D=SPACE_DIM)

    url = f'{model_urls_base["coordinates_colors"]}/{STATE_DICT_PREFIX}{epoch}{STATE_DICT_SUFIX}'

    state_dict = torch.hub.load_state_dict_from_url(model_urls['coordinates_colors'],
                                                    progress=progress)

    model.load_state_dict(state_dict)
    return model


def get_minkunet34c(progress=True, **kwargs):
    '''
    Returns the model MinkUNet34C from Choey et al.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
        **kwargs (dict): Keyword arguments that are passed on to MinkUNet34C. For details, see
            https://github.com/StanfordVL/MinkowskiEngine/blob/master/examples/minkunet.py and
            https://stanfordvl.github.io/MinkowskiEngine/demo/interop.html

    '''
    return MinkUNet34C(**kwargs)
