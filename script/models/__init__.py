from models.transformer import Transformer
from models.rnn import RNN

def get_model(name: str, out_dim: int, **kwargs):
    name = name.lower()
    if name == 'transformer':
        return Transformer(num_classes=out_dim, **kwargs)
    elif name == 'rnn':
        return RNN(num_classes=out_dim, **kwargs)
    else:
        raise NotImplementedError