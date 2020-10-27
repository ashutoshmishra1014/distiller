import torch
import torch.nn.functional as F
from torch import nn


__all__ = ['ASC']


model_urls = {
    'asc': './usecase_pretrained_models/asc_weights.pth',
}


class ASC(nn.Module):

    def __init__(self, num_classes=15):
        """

        Parameters
        ----------
        n_classes: int
            Number of classes. Default: 8
        """
        super(ASC, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)

        self.linear1 = nn.Linear(in_features=128, out_features=100)
        self.linear2 = nn.Linear(in_features=100, out_features=num_classes)

        self.dropout = nn.Dropout2d(p=0.3)

        layers = [self.conv1, self.conv2, self.linear1, self.linear2]
        self._layers = layers


    def forward(self, x):
        # x.shape = (batch_size, 1, 40, 500)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=5, stride=5)

        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(4, 100), stride=(4, 100))

        x = self.dropout(x)

        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, 128)

        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)  # softmax activation

        return x

    def predict(self, input, **kwargs):
        self.eval()
        with torch.no_grad():
            return self.forward(input)



def asc(pretrained=False):
    r"""Accoustic Scene Classification model
    Args:
        pretrained (bool): If True, returns a model pre-trained on DCase
    """
    model = ASC()
    if pretrained:
        model.load_state_dict(torch.load(model_urls['asc']))
    
    return model