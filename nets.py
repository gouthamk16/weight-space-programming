import torch


def load(path):
    """Rebuild a net from its state dict as a Sequential of Linear layers with ReLU in between.

    Rounds 1-2 saved a bare Linear (keys `weight`, `bias`); round 3 saved Linear-ReLU-Linear.
    """
    sd = torch.load(path, weights_only=True)
    if "weight" in sd:
        sd = {f"0.{k}": v for k, v in sd.items()}
    layers = []
    for k in sorted(k for k in sd if k.endswith("weight")):
        out, inp = sd[k].shape
        layers += [torch.nn.Linear(inp, out), torch.nn.ReLU()]
    net = torch.nn.Sequential(*layers[:-1])
    net.load_state_dict(sd)
    return net
