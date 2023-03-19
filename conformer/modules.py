from torch import nn


# 卷积层
class ConvolutionalLayer(nn.Module):
    def __init__(
            self,
            # Convolutional parameter
            in_channels: int = 100,
            out_channels: int = 10,
            kernel_size=(5, 5),
            # BN parameter
            num_features=(5, 4),
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device: bool = None,
            dtype=None,
            # ReLU parameter
            inplace: bool = True,

    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum,
                           affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class MaxpoolLayer(nn.Module):
    def __init__(
            self,
            kernel_size=(5, 4)
    ) -> None:
        super().__init__()
        self.maxpl = nn.MaxPool2d(kernel_size=kernel_size)

    def forward(self, x):
        x = self.maxpl(x)
        return x


class LinearLayer(nn.Module):

    def __init__(
            self,
            is_activation_function: bool = False,
            in_features: int = 256,
            out_features: int = 128,
            bias: bool = True) -> None:
        super().__init__()
        if not is_activation_function:
            self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
            self.tanh = nn.Tanh()

    def forwared(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        return x
