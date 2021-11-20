import torch.nn.functional as f
import torch.nn.init as init
import torch.nn as nn
import torch as t


class bconv2D(nn.Module):
    # baysian 2D convolutional layer
    def __init__(self, in_channels, out_channels,
                 kernel_size, prior_mu=0, prior_sigma=.5,
                 bais=True):
        super(bconv2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels, self.bais = out_channels, bais
        self.kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
        self.prior_mu, self.prior_sigma = prior_mu, prior_sigma

        self.weight_sigma = nn.Parameter(t.Tensor(out_channels,
                                    in_channels, *self.kernel_size),
                                         )
        self.weight_mu = nn.Parameter(t.Tensor(out_channels,
                                    in_channels, *self.kernel_size),
                                      )

        if bais:
            self.bais_sigma = nn.Parameter(t.Tensor(out_channels))
            self.bais_mu = nn.Parameter(t.Tensor(out_channels))
        else:
            self.register_parameter('bais_sigma', None)
            self.register_parameter('bais_mu', None)
        self.register_buffer('Sampfreeze', None)
        self.register_buffer('fweight_seed', None)
        self.register_buffer('fbais_seed', None)

        self.reset_parameters()

    @t.no_grad()
    def reset_parameters(self):
        init.xavier_normal_(self.weight_sigma)
        init.xavier_normal_(self.weight_mu)
        if self.bais:
            init.uniform_(self.bais_mu)
            init.uniform_(self.bais_sigma)

    def get_weight(self):
        bais = None
        if self.Sampfreeze is None:
            weight = self.weight_mu + t.randn_like(self.weight_sigma
                                                   ) * self.weight_sigma
            if self.bais:
                bais = self.bais_mu + t.randn_like(self.bais_sigma
                                                   ) * self.bais_sigma
        else:
            weight = self.weight_mu + self.fweight_seed * self.weight_sigma
            if self.bais:
                bais = self.bais_mu + self.fbais_seed * self.bais_sigma

        return weight, bais

    def freeze(self):
        self.Sampfreeze = True
        self.fweight_seed = t.randn_like(self.weight_sigma)
        self.fbais_seed = t.randn_like(self.bais_sigma)

    def unfreeze(self):
        self.Sampfreeze = None
        self.fweight_seed = None
        self.fbais_seed = None

    def forward(self, X):
        W, b = self.get_weight()
        #print(W.shape)
        #print(b.shape)
        return f.conv2d(X, W, b)


if __name__ == '__main__':
    bconv = bconv2D(3, 30, 3, 0, 0,False)
    tens = t.Tensor(10,3,50,50)
    out = bconv(tens)

