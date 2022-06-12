import torch
import torchvision

# Global dataset
# training_data = torchvision.datasets.ImageNet('../../data/images', split='train', download=True,
#                                               transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
#vgg16_true = torchvision.models.vgg16(pretrained=True)

# add layers in different position
# vgg16_false.add_module('add_linear_1', torch.nn.Linear(1000, 10))
# vgg16_false.classifier.add_module('add_linear_2', torch.nn.Linear(1000, 10))

# change a layer
#vgg16_false.classifier[6] = torch.nn.Linear(4096, 10)

# model save method 1 - save model and parameters
# save a model
# torch.save(vgg16_false, '../../pretrained_models/vgg16_m1.pth')
# vgg16 = torch.load('../../pretrained_models/vgg16_m1.pth')
# print(vgg16)

# model save method 2 - save parameters only (officially recommend) -> save storage space
# torch.save(vgg16_false.state_dict(), '../../pretrained_models/vgg16_method2.pth')
# model = torch.load('../../pretrained_models/vgg16_method2.pth')
# print(model)

# restore model structure
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('../../pretrained_models/vgg16_method2.pth'))
print(vgg16)
