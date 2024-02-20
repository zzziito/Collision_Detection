# from models.transformer import Transformer
from models.rnn import RNN
from models.cnn import CNN
from models.transformer import Transformer


def get_model(name: str, **kwargs):
    name = name.lower()
    if name == 'rnn':
        return RNN(num_joints=kwargs['num_joints'], hidden_size=kwargs['hidden_size'], num_layers=kwargs['num_layers'])
    elif name == 'cnn':
        return CNN(num_joints=kwargs['num_joints'], max_seq_len=kwargs['max_seq_len'])
    elif name == 'transformer':
        return Transformer(num_joints=kwargs['num_joints'], hidden_size=kwargs['hidden_size'], nhead=kwargs['nhead'], num_encoder_layers=kwargs['num_encoder_layers'])
    else:
        raise NotImplementedError