import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Define dataset
# dataset = torchvision.datasets.CIFAR10('../../data/images', train=False, transform=torchvision.transforms.ToTensor())
# dataloader = DataLoader(dataset, batch_size=16)
tr_data = torchvision.datasets.CIFAR10('../../data/images', train=True, transform=torchvision.transforms.ToTensor())
te_data = torchvision.datasets.CIFAR10('../../data/images', train=False, transform=torchvision.transforms.ToTensor())
print(f'The length of training data is {len(tr_data)}')
print(f'The length of test data is {len(te_data)}')

tr_data_loader = DataLoader(tr_data, batch_size=64)
te_data_loader = DataLoader(te_data, batch_size=64)


class Cifar10QuickModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1)
        self.max_pooling_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1)
        self.max_pooling_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1)
        self.max_pooling_3 = torch.nn.MaxPool2d(kernel_size=2)
        self.flatten = torch.nn.Flatten()
        self.full_connect_1 = torch.nn.Linear(64 * 4 * 4, 64)
        self.full_connect_2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pooling_1(x)
        x = self.conv2(x)
        x = self.max_pooling_2(x)
        x = self.conv3(x)
        x = self.max_pooling_3(x)
        x = self.flatten(x)
        x = self.full_connect_1(x)
        x = self.full_connect_2(x)
        return x


if __name__ == '__main__':

    # Test your network
    # x = torch.ones((64, 3, 32, 32))
    # model = Cifar10QuickModel()
    # print(model(x).shape)

    # Visualize in TensorBoard
    # writer = SummaryWriter('../../logs')
    # writer.add_graph(model, x)
    # writer.close()

    # Training
    model = Cifar10QuickModel()

    if torch.cuda.is_available():
        model = model.cuda()

    loss_function = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    writer = SummaryWriter('../../logs')

    for epoch in range(20):
        epoch_loss = 0
        for data in tr_data_loader:
            images, targets = data
            y = model(images)
            loss = loss_function(y, targets)

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss = epoch_loss + loss
        print(f'Iteration {epoch}: loss is {epoch_loss}')
        writer.add_scalar('training_loss', epoch_loss, epoch)

        # test
        total_test_loss = 0
        for data in te_data:
            image, target = data
            y = model(image)
            loss = loss_function(y, target)
            total_test_loss = total_test_loss + loss
        writer.add_scalar('test_loss', total_test_loss, epoch)

