import torch
import torchvision
from torch import nn, optim
def resnet_50_4_channel(PATH):
    """
    Function for fetching custom resnet model
    """
    model = torchvision.models.resnet50()
    model.conv1 = nn.Conv2d(4, 64, (7, 7),
                            stride=(2,2),
                            padding=(3,3),
                            bias=False)

    # output features to 3 classification
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=2048,
            out_features=19
        ),
        torch.nn.Sigmoid())

    model.load_state_dict(torch.load(PATH))
    return(model)
