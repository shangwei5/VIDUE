import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.blocks import Sty_layer, CA_layer

class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """


        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
           
    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """


        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x
    
class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        
        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
           
    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, size=[skpCn.size(2), skpCn.size(3)], mode='bilinear')
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope = 0.1)
        return x



class UNet2(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.
    
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        
        super(UNet2, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1   = up(512, 512)
        self.up2   = up(512, 256)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)
        
    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """


        x  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x  = self.down5(s5)
        x  = self.up1(x, s5)
        x  = self.up2(x, s4)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x1  = self.up5(x, s1)
        x  = F.leaky_relu(self.conv3(x1), negative_slope = 0.1)
        return x


class UNet2_V(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        super(UNet2_V, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

        self.dse3 = CA_layer(256, 512, 4)
        self.dse2 = CA_layer(256, 256, 4)
        self.dse1 = CA_layer(256, 64, 4)
        self.use2 = CA_layer(256, 256, 4)
        self.use1 = CA_layer(256, 64, 4)

    def forward(self, x, vec):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s2 = self.dse1(s2, vec)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s4 = self.dse2(s4, vec)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.dse3(x, vec)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.use2(x, vec)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.use1(x, vec)
        x1 = self.up5(x, s1)
        x = F.leaky_relu(self.conv3(x1), negative_slope=0.1)
        # x = self.conv3(x1)
        return x


class UNet2_V2(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels, vecChannels=256):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        super(UNet2_V2, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

        # self.dse3 = CA_layer(256, 512, 4)
        # self.dse2 = CA_layer(256, 256, 4)
        # self.dse1 = CA_layer(256, 64, 4)
        if vecChannels == 256:
            reduc = 4
        else:
            reduc = 1
        self.use5 = CA_layer(vecChannels, 32, reduc)
        self.use4 = CA_layer(vecChannels, 64, reduc)
        self.use3 = CA_layer(vecChannels, 128, reduc)
        self.use2 = CA_layer(vecChannels, 256, reduc)
        self.use1 = CA_layer(vecChannels, 512, reduc)

    def forward(self, x, vec):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        # s2 = self.dse1(s2, vec)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        # s4 = self.dse2(s4, vec)
        s5 = self.down4(s4)
        x = self.down5(s5)
        # x = self.dse3(x, vec)
        x = self.up1(x, s5)
        x = self.use1(x, vec)
        x = self.up2(x, s4)
        x = self.use2(x, vec)
        x = self.up3(x, s3)
        x = self.use3(x, vec)
        x = self.up4(x, s2)
        x = self.use4(x, vec)
        x = self.up5(x, s1)
        x1 = self.use5(x, vec)
        x = F.leaky_relu(self.conv3(x1), negative_slope=0.1)
        # x = self.conv3(x1)
        return x

class UNet2_V3(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        super(UNet2_V3, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

        # self.dse3 = CA_layer(256, 512, 4)
        # self.dse2 = CA_layer(256, 256, 4)
        # self.dse1 = CA_layer(256, 64, 4)
        self.use5 = Sty_layer(256, 32, 4)
        self.use4 = Sty_layer(256, 64, 4)
        self.use3 = Sty_layer(256, 128, 4)
        self.use2 = Sty_layer(256, 256, 4)
        self.use1 = Sty_layer(256, 512, 4)

    def forward(self, x, vec):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        # s2 = self.dse1(s2, vec)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        # s4 = self.dse2(s4, vec)
        s5 = self.down4(s4)
        x = self.down5(s5)
        # x = self.dse3(x, vec)
        x = self.up1(x, s5)
        x = self.use1(x, vec)
        x = self.up2(x, s4)
        x = self.use2(x, vec)
        x = self.up3(x, s3)
        x = self.use3(x, vec)
        x = self.up4(x, s2)
        x = self.use4(x, vec)
        x = self.up5(x, s1)
        x1 = self.use5(x, vec)
        x = F.leaky_relu(self.conv3(x1), negative_slope=0.1)
        # x = self.conv3(x1)
        return x

