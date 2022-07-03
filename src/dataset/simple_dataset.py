import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):

    def __init__(self, data, label):
        self.data, self.label = torch.from_numpy(data), torch.from_numpy(label)
        self.data, self.label = self.data.type(torch.FloatTensor), self.label.type(torch.FloatTensor)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('../../data/SZ000001.csv')
    X = df.iloc[0:-1, 2:].values        # Data of T
    y = df.iloc[1:, 2].values           # Open price of T+1
    ds = SimpleDataset(X, y)

    from torch.utils.data import DataLoader
    ds_loader = DataLoader(ds, batch_size=128, shuffle=True)

    print(len(ds))
