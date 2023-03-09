import torch
import torch.nn as nn
from torch import relu, sigmoid


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

        self.conv1 = nn.Conv2d(2, 4, kernel_size=kernelSize, stride=stride)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=kernelSize, stride=stride)

        convw = self._conv2dSizeOut(self._conv2dSizeOut(arrayWidth))
        convh = self._conv2dSizeOut(self._conv2dSizeOut(arrayHeight))
        self.linear1 = nn.Linear(convw * convh * 8, 32)
        self.linear2 = nn.Linear(32, numOutputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x.to(self.device)
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))

        return sigmoid(self.linear2(relu(self.linear1(x.view(x.size(0), -1)))))

    def _conv2dSizeOut(self, size: int):
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        return (size - (self.kernelSize - 1)) // self.stride
