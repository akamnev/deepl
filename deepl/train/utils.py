
def kld_loss(module, method_name, method_kwargs=None):
    if method_kwargs is None:
        method_kwargs = {}
    loss = 0.0
    if hasattr(module, method_name):
        loss += getattr(module, method_name)(**method_kwargs)
    for child in module.children():
        loss += kld_loss(child, method_name, method_kwargs)
    return loss
