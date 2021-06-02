import torch
from functools import partial


def init_weights(module, initializer_range):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data.normal_(
            mean=0.0, std=initializer_range)
    if isinstance(module, torch.nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def tie_weights(self):
    """
    Tie the weights between the input embeddings and the output embeddings.
    """
    output_embeddings = self.get_output_embeddings()
    input_embeddings = self.get_input_embeddings()
    output_embeddings.weight = input_embeddings.weight

    if getattr(output_embeddings, "bias", None) is not None:
        output_embeddings.bias.data = torch.nn.functional.pad(
            output_embeddings.bias.data,
            (0, output_embeddings.weight.shape[0]
                - output_embeddings.bias.shape[0], ),
            "constant",
            0,
        )
    if hasattr(output_embeddings, "out_features") and \
            hasattr(input_embeddings, "num_embeddings"):
        output_embeddings.out_features = input_embeddings.num_embeddings

