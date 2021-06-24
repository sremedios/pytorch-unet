import torch

from resize.pytorch import resize


class _ConvBlock(torch.nn.Sequential):
    """Abstract class of the conv block.

    Attributes:
        in_channels (int): The number of channels of the input.
        out_channels (int): The number of channels of the output.

    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = self._create_conv()
        self.norm = self._create_norm()
        self.activ = self._create_activ()
        self.dropout = self._create_dropout()

    def _create_conv(self):
        raise NotImplementedError

    def _create_norm(self):
        raise NotImplementedError

    def _create_activ(self):
        raise NotImplementedError

    def _create_dropout(self):
        raise NotImplementedError


class ConvBlock(_ConvBlock):
    """Convolution block used in UNet.

    """
    def _create_conv(self):
        return torch.nn.Conv3d(self.in_channels, self.out_channels, 3,
                               bias=False, padding=1)

    def _create_activ(self):
        return torch.nn.ReLU()

    def _create_norm(self):
        return torch.nn.InstanceNorm3d(self.out_channels, affine=True)

    def _create_dropout(self):
        return torch.nn.Identity()


class ProjBlock(ConvBlock):
    """Convolution block with kernel size 1.

    """
    def _create_conv(self):
        return torch.nn.Conv3d(self.in_channels, self.out_channels, 1,
                               bias=False)


class _ContractingBlock(torch.nn.Sequential):
    """Abstract class of the contracting block.

    Attributes:
        in_channels (int): The number of channels of the input.
        out_channels (int): The number of channels of the output.
        mid_channels (int): The number of channels of the intermediate tensor.

    """
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.conv0 = self._create_conv0()
        self.conv1 = self._create_conv1()

    def _create_conv0(self):
        raise NotImplementedError

    def _create_conv1(self):
        raise NotImplementedError


class ContractingBlock(_ContractingBlock):
    """The contracting block of the UNet.

    """
    def _create_conv0(self):
        return ConvBlock(self.in_channels, self.mid_channels)

    def _create_conv1(self):
        return ConvBlock(self.mid_channels, self.out_channels)


class _ExpandingBlock(torch.nn.Module):
    """The abstract class of the expanding block.

    Attributes:
        in_channels (int): The number of the channels of the input.
        shortcut_channels (int): The number of the channels of the shortchut.
        out_channels (int): The number of channels of the output.
        conv0 (torch.nn.Module): The first convolution block.
        conv1 (torch.nn.Module): The second convolution block.

    """
    def __init__(self, in_channels, shortcut_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut_channels = shortcut_channels
        self.conv0 = self._create_conv0()
        self.conv1 = self._create_conv1()

    def forward(self, input, shortcut):
        output = torch.cat((input, shortcut), dim=1) # concat channels
        output = self.conv0(output)
        output = self.conv1(output)
        return output

    def _create_conv0(self):
        raise NotImplementedError

    def _create_conv1(self):
        raise NotImplementedError


class ExpandingBlock(_ExpandingBlock):
    """The expanding block of the UNet.

    """
    def _create_conv0(self):
        in_channels = self.in_channels + self.shortcut_channels
        return ConvBlock(in_channels, self.out_channels)

    def _create_conv1(self):
        return ConvBlock(self.out_channels, self.out_channels)


class _TransUpBlock(torch.nn.Sequential):
    """The abstract class of the transition up block.

    Attributes:
        in_channels (int): The number of the channels of the input.
        out_channels (int): The number of the channels of the output.

    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = self._create_conv()
        self.up = self._create_up()

    def _create_conv(self):
        raise NotImplementedError

    def _create_up(self):
        raise NotImplementedError


class TransUpBlock(_TransUpBlock):
    """The transition block of the UNet.

    """
    def _create_conv(self):
        return ProjBlock(self.in_channels, self.out_channels)

    def _create_up(self):
        return Resize3d(scale_factor=2)


class Resize3d(torch.nn.Module):
    def __init__(self, scale_factor=2, order=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.order = order
        self._dxyz = (1 / self.scale_factor, ) * 3

    def forward(self, image):
        return resize(image, self._dxyz, order=self.order)
