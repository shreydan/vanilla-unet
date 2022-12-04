import torch
import torch.nn as nn
import torchvision.transforms.functional as vt


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, crop=True) -> None:

        super(UNet, self).__init__()

        self.channel_list = [64, 128, 256, 512]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.crop = crop

        self.pooler = nn.MaxPool2d(2)

        self.downs = nn.ModuleList()
        for channel in self.channel_list:
            self.downs.append(self._double_conv(in_channels, channel))
            in_channels = channel

        self.latent = self._double_conv(in_channels, in_channels * 2)

        self.ups = nn.ModuleList()
        for channel in reversed(self.channel_list):
            self.ups.extend(
                [
                    nn.ConvTranspose2d(channel * 2, channel, 2, 2),
                    self._double_conv(channel * 2, channel),
                ]
            )

        self.output = nn.Conv2d(self.channel_list[0], self.num_classes, 1, 1)

    def _double_conv(self, in_channels, out_channels) -> nn.Sequential:

        padding = 0 if self.crop else 1

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=padding),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=padding),
            nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:

        skip_connections = []

        for layer in self.downs:
            x = layer(x)
            skip_connections.append(x)
            x = self.pooler(x)

        x = self.latent(x)

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):

            up_conv = self.ups[i]
            x = up_conv(x)

            skip = skip_connections[i // 2]

            if self.crop:
                skip = vt.center_crop(skip, output_size=x.shape[2:])
            else:
                x = vt.center_crop(x, output_size=skip.shape[2:])

            x = torch.concat([skip, x], dim=1)

            double_conv = self.ups[i + 1]
            x = double_conv(x)

        x = self.output(x)

        return x


if __name__ == "__main__":

    x = torch.rand(1, 1, 572, 572)  # batch x channels x height x width
    model = UNet(in_channels=1, num_classes=2, crop=False)
    print(model(x).shape)

    """
      crop enabled => expected: B x C x H x W : 1 x 2 x 388 x 388
     crop disabled => expected: B x C x H x W : 1 x 2 x 572 x 572

    """
