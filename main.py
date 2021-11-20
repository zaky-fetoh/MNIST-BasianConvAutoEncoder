import torch as t
import torch.nn as nn
import torch.optim as optim

import model as m
import train as tr
import data_org as dorg
import visualize as v



def getloss_fun(loss, klweight=.2, lsweight=.8):
    def loss_fun(pred, act, net):
        return lsweight * loss(pred, act) + klweight * m.Baysian_KL_Loss(net)

    return loss_fun


class main:
    def __init__(self):
        self.net = m.BAE().cuda()
        # self.bce_loss = nn.BCEWithLogitsLoss()
        # self.loss_fn = getloss_fun(self.bce_loss)
        self.loss_fn = getloss_fun(m.ssim)
        self.opt = optim.Adam(self.net.parameters())
        self.trloader, self.valoader = dorg.getloaders()

    def train(self, epochs=20):
        self.tloss, self.vloss = tr.train_validate(self.net,
                                                   self.trloader, self.valoader, epochs,
                                                   self.loss_fn, self.opt,
                                                   )

    def encode(self, imgs):
        return self.net.encoder(imgs)

    def decode(self, codes):
        return self.net.decoder(codes)

    def load_net(self, name, ):
        tr.load_model(name, self.net)

    def plot_trset(self, show=True):
        v.plotset(self.trloader, self.net, show=show)

    def laten_space_evol(self, models_num=20):
        for n in range(models_num):
            self.load_net('weight/weight' + str(n) + '.pth')
            self.plot_trset(show=False);
            v.plt.savefig('fig' + str(n) + '.png')


if __name__ == '__main__':
    t.cuda.empty_cache()
    model = main()
    model.train()
    # model.load_net('weight/weight11.pth')
