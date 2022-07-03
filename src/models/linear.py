import pandas as pd
import torch.nn as nn
import torch.nn.functional

from torch.utils.data import DataLoader
from dataset.simple_dataset import SimpleDataset


class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
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

    # Training
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(lm.parameters(), lr=0.001)

    lm.eval()
    total_loss = 0
    for epoch in range(100):
        lm.train()
        for x, y in ds:
            optimizer.zero_grad()
            pred = lm(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = lm(x)
                loss = loss_fn(pred, y)

        print(loss)



    # print model parameters
    print('\n\nJust one layer:')
    print(lm.model[0].weight)

