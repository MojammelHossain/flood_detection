import torch
from torchvision import models
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

class LayerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.relu1 = nn.ReLU()

        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        return self.relu2(self.conv2(self.drop1(self.relu1(self.conv1(x)))))

class Encoder(nn.Module):
    def __init__(self, channels=[3, 16, 32, 64, 128]):
        super().__init__()
        self.encoder = nn.ModuleList(
            LayerBlock(channels[i], channels[i+1]) 
            for i in range(len(channels)-1)
        )
        self.maxpl = nn.MaxPool2d(2)
    
    def forward(self, x):
        encFea = []
        for encBlock in self.encoder:
            x = encBlock(x)
            encFea.append(x)
            x = self.maxpl(x)
        
        return encFea

class Decoder(nn.Module):
    def __init__(self, channels=[256, 128, 64, 32, 16]):
        super().__init__()
        self.layers = len(channels)-1
        self.convTrans = nn.ModuleList(
            nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2)
            for i in range(len(channels)-1)
        )
        self.decoder = nn.ModuleList(
            LayerBlock(channels[i], channels[i+1])
            for i in range(len(channels)-1)
        )
    
    def forward(self, x, encFea):
        for i in range(self.layers):
            x = self.convTrans[i](x)
            x = torch.cat([x, encFea[i]], dim=1)
            x = self.decoder[i](x)
        return x

class UNET(nn.Module):
    def __init__(self, enc_channels, dec_channels, num_classes):
        super().__init__()
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.num_classes = num_classes
        self.encoder = Encoder(self.enc_channels)
        self.decoder = Decoder(self.dec_channels)
        self.maxpl = nn.MaxPool2d(2)

        self.midblock = LayerBlock(self.enc_channels[-1], self.dec_channels[0])

        self.outConv = nn.Conv2d(dec_channels[-1], self.num_classes, kernel_size=1)
        self.outsoft = nn.Softmax(dim=1)
    
    def forward(self, x):
        encOut = self.encoder(x)

        midOut = self.midblock(self.maxpl(encOut[-1]))

        decOut = self.decoder(midOut, encOut[::-1])
        return self.outsoft(self.outConv(decOut))

if __name__ == '__main__':
    model = UNET([2,16,32,64,128], [256,128,64,32,16], 2)
    model = model.to(device="cuda", memory_format=torch.channels_last)
    