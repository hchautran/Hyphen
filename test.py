import torch


def test():
    a = torch.ones((1, 30))
    b = torch.ones((1, 30))
    c = a + b
    return c


if __name__ == "__main__":
    print(test())
