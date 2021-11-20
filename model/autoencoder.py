import torch as t
import torch.nn as nn
import torch.nn.functional as f
from model import *


class encoder_block(nn.Module):
    def __init__(self, in_chennel, out_channel,
                 kernel, dropout=.5,
                 maxpool=True, actfunc=f.relu,
                 fx = bconv2D
                 ):
        super(encoder_block, self).__init__()
        mlis = []
        self.actfunc = actfunc
        if dropout:
            mlis.append(nn.Dropout2d(dropout))
        mlis += [
            fx(in_chennel, out_channel, kernel),
            nn.BatchNorm2d(out_channel),
        ]
        self.maxpool = maxpool
        self.layers = nn.Sequential(*mlis)

    def forward(self, X):
        #print(X.shape)
        X = self.layers(X)
        if self.maxpool:
            X = f.max_pool2d(X, 2)
        return self.actfunc(X)


class encoder(nn.Module):
    # an Encoder module of the BAE
    def __init__(self, input_channels=1,
                 layer_sizes=[64, 128, 256, 256, 2],
                 kernel_sizes=[3] * 5,
                 dropout=[0] * 5,
                 actfunc=[f.leaky_relu] * 4+ [lambda x:x],
                 maxpool=[True]  + [False] * 4
                 ):
        super(encoder, self).__init__()
        inps = input_channels
        lis = list()
        for i, ls in enumerate(layer_sizes):
            lis.append(encoder_block(
                inps, layer_sizes[i], kernel_sizes[i],
                dropout[i], maxpool[i],
                actfunc[i],
            ))
            inps = layer_sizes[i]
        self.blocks = nn.Sequential(*lis)

    def forward(self, X):
        X = self.blocks(X)
        return f.avg_pool2d(X, tuple(X.shape[-2:]))  # .view(X.size(0), -1 )


class decoder_block(nn.Module):
    # decoder block
    def __init__(self, in_channel, out_channel, kernel,
                 upool=2, dropout=0, actfunc=f.relu,
                 fx = bconv2D):
        super(decoder_block, self).__init__()
        lis = list()
        if dropout:
            lis.append(nn.Dropout2d(dropout))
        if upool:
            lis.append(nn.UpsamplingBilinear2d(scale_factor=upool))
        lis.append(fx(in_channel, out_channel, kernel))
        lis.append(nn.BatchNorm2d(out_channel))
        self.actfunc = actfunc
        self.layers = nn.Sequential(*lis)

    def forward(self, X):
        #print(X.shape)
        X = self.layers(X)
        return self.actfunc(X)


class decoder(nn.Module):
    # decoder module
    def __init__(self, inps=2,
                 layers=[32, 64, 128, 256, 1],
                 upscale=[6,3,0,2,2],
                 kernel=[3] * 5,
                 dropout=[0] * 5,
                 actfunc=[f.leaky_relu] * 4+[f.sigmoid],
                 ):
        super(decoder, self).__init__()
        lis = list()
        for i in range(layers.__len__()):
            lis.append(
                decoder_block(inps, layers[i], kernel[i],
                              upscale[i], dropout[i], actfunc[i],
                              ))
            inps= layers[i]
        self.layers = nn.Sequential(*lis)

    def forward(self, X):
        X = self.layers(X)
        return f.pad(X,[1,1,1,1])

class BAE(nn.Module):
    def __init__(self, encoder= encoder(),
                 decoder =decoder()  ):
        super(BAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, X):
        X = self.encoder(X)
        return self.decoder(X)



if __name__ == '__main__':
    enc = encoder()
    deco= decoder()
    for m in enc.modules():
        if isinstance(m, bconv2D):
            print("ok")
    tens = t.Tensor(10, 1, 50, 50)
    encd = enc(tens)
    decd = deco(encd)
    print(decd.shape)
