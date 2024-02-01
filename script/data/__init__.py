from data.robros_rnn import RobrosRNN
from data.robros_cnn import RobrosCNN

def get_dataloader(name: str, **kwargs):
    name = name.lower()
    if name == 'rnn':
        return RobrosRNN(**kwargs)
    elif name == 'cnn':
        return RobrosCNN(**kwargs)
    else:
        raise NotImplementedError