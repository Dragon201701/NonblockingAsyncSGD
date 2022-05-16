import torch

def serialize_parameters(model, grad=False):
    _params = torch.Tensor()
    for param in model.parameters():
        if grad:
            _params = torch.cat((_params, param.grad.view(-1)))
        else:
            _params = torch.cat((_params, param.data.view(-1)))
    return _params

def deserialize_parameters(model, parameters, grad=False):
    idx = 0
    for param in model.parameters():
        n = param.data.numel()
        size = param.data.size()
        if grad:    
            param.grad = parameters[idx:idx+n].view(size).clone()
        else:
            param.data.copy_(parameters[idx:idx+n].view(size))
        idx += n