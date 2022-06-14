import torch
import torch.nn as nn
from torch.nn import Parameter


class DummyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.linear.weight = Parameter(torch.tensor([2.0, 3.0]))
        self.linear.bias = Parameter(torch.tensor(5.0))
        self.relu = torch.nn.ReLU()

    def forward(self, x) -> torch.tensor:
        y = self.linear(x)
        return y


if __name__ == '__main__':
    dm = DummyModel()
    print(dm.eval())
    x = torch.tensor([1.0, 2.0])
    print(dm(x))
    torch.save(dm, '../../pretrained_models/dummy_model.pth')
