import os
import torch
import torch.nn as nn
from .config import ConfigBase


class ModelBase(nn.Module):
    config_cls = ConfigBase

    def __init__(self, config):
        super().__init__()
        if not isinstance(config, self.config_cls):
            raise ValueError(
                f'Parameter config in `{self.__class__.__name__}(config)` '
                f'should be an instance of '
                f'class `{self.config_cls.__class__.__name__}`.'
            )
        self.config = config

    def save(self, save_directory=None, model_name='model'):
        if save_directory is None:
            save_directory = os.path.curdir
        assert os.path.isdir(
            save_directory
        ), 'Saving path should be a directory where the model and ' \
           'configuration can be saved'
        output_model_file = os.path.join(save_directory, model_name + '.bin')
        output_config_file = os.path.join(save_directory, model_name + '.json')
        torch.save(self.state_dict(), output_model_file)
        self.config.to_file(output_config_file)

    @classmethod
    def load(cls, load_directory=None, model_name='model',
             init_weights=False, **kwargs):
        if load_directory is None:
            load_directory = os.path.curdir
        input_model_file = os.path.join(load_directory, model_name + '.bin')
        input_config_file = os.path.join(load_directory, model_name + '.json')
        config = cls.config_cls.from_file(input_config_file, **kwargs)
        obj = cls(config)
        if init_weights:
            obj.init_weights()
        obj.load_state_dict(torch.load(input_model_file, map_location='cpu'))
        return obj

    def num_parameters(self, only_trainable=False):
        params = filter(lambda x: x.requires_grad,
                        self.parameters()) if only_trainable else self.parameters()
        return sum(p.numel() for p in params)
