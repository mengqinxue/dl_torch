import pandas as pd
import torch.nn as nn
import torch.nn.functional

from torch.utils.data import DataLoader
from src.dataset.simple_dataset import SimpleDataset


class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Softmax(10),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    df = pd.read_csv('../../data/SZ000001.csv')
    X = df.iloc[0:-1, 2:].values  # Data of T
    y = df.iloc[1:, 2].values     # Open price of T+1
    ds = SimpleDataset(X, y)

    ds_loader = DataLoader(ds, batch_size=128, shuffle=True)

    lm = LinearModel()

    print('The model:')
    print(lm)

    print('\n\nJust one layer:')
    print(lm.model[0].weight)

