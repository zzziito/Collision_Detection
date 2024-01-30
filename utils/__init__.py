import torch
from torch import nn
from typing import Union, Dict

from utils import losses, compute

def clone_state_dict(thing: Union[nn.Module, Dict[str, torch.Tensor]]):
    if isinstance(thing, nn.Module):
        state_dict = thing.state_dict()
    elif isinstance(thing, dict):
        state_dict = thing
    else:
        raise TypeError(f"Expected `nn.Module` or `dict[str, torch.Tensor]` for `thing` but got `{repr(thing)}` instead.")
    
    return {key: val.clone().detach().cpu() for key, val in state_dict.items()}