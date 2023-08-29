import torch
from hyptorch.models.clip import HypCLIPTextTransformer


def test():
    a = torch.ones((1, 30))
    b = torch.ones((1, 30))
    c = a + b
    return c


if __name__ == "__main__":
    

