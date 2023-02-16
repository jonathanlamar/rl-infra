import torch
import torch.nn as nn
from torch.nn.functional import relu


class DeepQNetwork(nn.Module):
    kernelSize: int
    stride: int
    device: torch.device

    def __init__(
        self,
        arrayHeight: int,
        arrayWidth: int,
        numOutputs: int,
        device: torch.device,
        kernelSize: int = 4,
        stride: int = 1,
    ) -> None:
        super(DeepQNetwork, self).__init__()
        self.kernelSize = kernelSize
        self.stride = stride
        self.device = device

        # TODO: Make this configurable
        self.conv1 = nn.Conv2d(1, 16, kernel_size=kernelSize, stride=stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernelSize, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=kernelSize, stride=stride)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=kernelSize, stride=stride)
        self.bn4 = nn.BatchNorm2d(32)

        convw = self._conv2dSizeOut(self._conv2dSizeOut(self._conv2dSizeOut(self._conv2dSizeOut(arrayWidth))))
        convh = self._conv2dSizeOut(self._conv2dSizeOut(self._conv2dSizeOut(self._conv2dSizeOut(arrayHeight))))
        self.linear1 = nn.Linear(convw * convh * 32, 16)
        self.linear2 = nn.Linear(16, 8)
        self.linear3 = nn.Linear(8, numOutputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x.to(self.device)
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))
        x = relu(self.bn4(self.conv4(x)))

        return self.linear3(self.linear2(self.linear1(x.view(x.size(0), -1))))

    def _conv2dSizeOut(self, size: int):
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        return (size - (self.kernelSize - 1)) // self.stride
