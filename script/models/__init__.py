

def get_model(name: str, out_dim: int, **kwargs):
    name = name.lower()
    if name == 'gae':
        return GeneralAutoencoder(num_classes=out_dim, **kwargs)
    else:
        return DomainSplittedModule(name, out_dim)