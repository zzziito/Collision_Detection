# from models.transformer import Transformer
from models.rnn import RNN
from models.cnn import CNN
from models.transformer import Transformer
# from models.transformer_wdecoder import Dtransformer
from models.transformer_new import Transformer_2
from models.discriminator_fc import Discriminator


def get_model(name: str, **kwargs):
    name = name.lower()
    if name == 'rnn':
        return RNN(num_joints=kwargs['num_joints'], hidden_size=kwargs['hidden_size'], num_layers=kwargs['num_layers'])
    elif name == 'cnn':
        return CNN(num_joints=kwargs['num_joints'], max_seq_len=kwargs['max_seq_len'])
    elif name == 'transformer':
        return Transformer_2(seq_len=1000, num_joints=kwargs['num_joints'], hidden_size=kwargs['hidden_size'], nhead=kwargs['nhead'], num_encoder_layers=kwargs['num_encoder_layers'])
    elif name == 'fc':
        return Discriminator(input_size=3000, hidden_size1=64, hidden_size2=32, output_size=3000)
    else:
        raise NotImplementedError