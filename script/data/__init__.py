from data.robros_free import RobrosRNN
# from data.robros_collision import RobrosRNN_Collision
# from data.robros_cnn import RobrosCNN
# from data.discriminator import RobrosDisc

def get_dataloader(name: str, **kwargs):
    name = name.lower()
    if name == 'rnn':
        return RobrosRNN(**kwargs)
    # elif name == 'cnn':
    #     return RobrosCNN(**kwargs)
    elif name == 'transformer':
        return RobrosRNN(**kwargs)
    # elif name == 'discriminator':
        # return RobrosRNN_Collision(**kwargs)
    else:
        raise NotImplementedError