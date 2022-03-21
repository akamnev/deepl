import torch.nn as nn


def kld_loss(module, method_name, method_kwargs=None):
    if method_kwargs is None:
        method_kwargs = {}
    loss = 0.0
    if hasattr(module, method_name):
        loss += getattr(module, method_name)(**method_kwargs)
    for child in module.children():
        loss += kld_loss(child, method_name, method_kwargs)
    return loss


def kld_loss_nan(module, method_name, method_kwargs=None):
    """Функция необходима для случая когда не все слои модели
    работают."""
    if method_kwargs is None:
        method_kwargs = {}
    loss = []
    if hasattr(module, method_name):
        try:
            loss += [getattr(module, method_name)(**method_kwargs)]
        except Exception:
            pass
    for child in module.children():
        loss += kld_loss_nan(child, method_name, method_kwargs)
    return loss


def reset_module_loss(module, method_name, value_name):
    if hasattr(module, method_name):
        setattr(module, value_name, None)
    for child in module.children():
        reset_module_loss(child, method_name, value_name)


def collect_layer_norm_weights(module):
    fval = []
    if isinstance(module, nn.LayerNorm):
        fval += [module.weight]
    for child in module.children():
        fval += collect_layer_norm_weights(child)
    return fval
