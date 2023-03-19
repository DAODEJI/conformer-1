from torch import nn
from conformer import ConformerBlock
import modules


class Module(nn.Module):
    def __init__(
            self,
            num_layers: int = 2,
            dim: int = 10) -> None:
        super().__init__()
        self.model = nn.Sequential(
            modules.ConvolutionalLayer(),
            modules.MaxpoolLayer(),
            modules.ConvolutionalLayer(),
            modules.MaxpoolLayer(),
            modules.ConvolutionalLayer(),
            modules.MaxpoolLayer(),
            nn.ModuleList(ConformerBlock(dim=dim) for _ in range(num_layers)),
            modules.LinearLayer(False),
            modules.LinearLayer(False),
            modules.LinearLayer(True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    test = Module()
    print(test)
