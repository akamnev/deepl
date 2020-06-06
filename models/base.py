import os
import torch
import torch.nn as nn
from utils.config import BERTConfig


class UtilsMixin:
    def num_parameters(self, only_trainable: bool = False) -> int:
        """Get number of (optionally, trainable) parameters in the module.
        """
        params = filter(lambda x: x.requires_grad,
                        self.parameters()) if only_trainable else self.parameters()
        return sum(p.numel() for p in params)


class BERTBase(nn.Module, UtilsMixin):
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, BERTConfig):
            raise ValueError(
                f'Parameter config in `{self.__class__.__name__}(config)` '
                f'should be an instance of class `BERTConfig`.'
            )
        self.config = config

    def init_weights(self):
        self.apply(self._init_weights)

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
    def load(cls, load_directory=None, model_name='model', init_weights=True):
        if load_directory is None:
            load_directory = os.path.curdir
        input_model_file = os.path.join(load_directory, model_name + '.bin')
        input_config_file = os.path.join(load_directory, model_name + '.json')
        config = BERTConfig.from_file(input_config_file)
        obj = cls(config)
        if init_weights:
            obj.init_weights()
        obj.load_state_dict(torch.load(input_model_file, map_location='cpu'))
        obj.to(config.device)
        return obj


class LMMixin:

    def get_input_embeddings(self):
        raise NotImplementedError

    def get_output_embeddings(self):
        raise NotImplementedError

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()
        output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0]
                    - output_embeddings.bias.shape[0], ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and \
                hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings
