import torch as t
import math
from model import *


def _klLoss(mu_0, log_sigma_0, mu_1, log_sigma_1):
    kl = log_sigma_1 - log_sigma_0 + \
         (t.exp(log_sigma_0) ** 2 + (mu_0 - mu_1) ** 2) / (2 * math.exp(log_sigma_1) ** 2) - 0.5
    return kl.sum()


def Baysian_KL_Loss(bnet:t.nn.Module):
    device = t.device('cuda' if next(bnet.parameters()).is_cuda else 'cpu')
    kl_acc = t.Tensor([0]).to(device=device)
    n = t.Tensor([1]).to(device=device)
    for m in bnet.modules():
        if isinstance(m, bconv2D ):
            kl_acc += _klLoss(m.weight_mu,m.weight_sigma,
                              m.prior_mu, m.prior_sigma)
            n += m.weight_sigma.view(-1).size()[0]
            if m.bais :
                kl_acc += _klLoss(m.bais_mu, m.bais_sigma,
                                  m.prior_mu, m.prior_sigma)
                n += m.bais_sigma.view(-1).size()[0]
    return kl_acc / n


if __name__ == '__main__':
    bnet = bconv2D(3,10,3)
    kloss = Baysian_KL_Loss(bnet)
    print(kloss)




