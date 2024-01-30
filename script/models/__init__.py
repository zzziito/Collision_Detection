from models.transformer import Transformer
from models.rnn import RNN
from models.cnn import CNN

def get_model(name: str, **kwargs):
    name = name.lower()
    if name == 'transformer':
        return Transformer(**kwargs)
    elif name == 'rnn':
        return RNN(**kwargs)
    elif name == 'cnn':
        return CNN(**kwargs)
    else:
        raise NotImplementedError