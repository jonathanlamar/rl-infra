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
        kernelSize: int = 5,
        stride: int = 2,
    ) -> None:
        super(DeepQNetwork, self).__init__()
        self.kernelSize = kernelSize
        self.stride = stride
        self.device = device

        # TODO: Make this configurable
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernelSize, stride=stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernelSize, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=kernelSize, stride=stride)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2dSizeOut(size: int):
            return (size - (kernelSize - 1) - 1) // stride + 1

        convw = conv2dSizeOut(conv2dSizeOut(conv2dSizeOut(arrayWidth)))
        convh = conv2dSizeOut(conv2dSizeOut(conv2dSizeOut(arrayHeight)))
        linearInputSize = convw * convh * 32
        self.head = nn.Linear(linearInputSize, numOutputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))
