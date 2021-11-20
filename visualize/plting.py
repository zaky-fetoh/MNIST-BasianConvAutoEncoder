import matplotlib.pyplot as plt
import numpy as np


@t.no_grad()
def plotset(loader, net, device='cuda', show=True):
    d = {i: list() for i in range(10)}
    for imgs, lbl in loader:
        imgs = imgs.to(device=device)
        lbl = lbl.cpu().view(-1).detach().numpy()
        codes = net.encoder(imgs).view(imgs.size(0), -1).cpu().detach().numpy()
        # print(codes.shape)
        for i in range(10):
            d[i].append(codes[lbl == i])
            # print(d[i][0].shape)
    for key, val in d.items():
        arr = np.concatenate(val, axis=0)
        plt.plot(arr[:, 0], arr[:, 1], '+', label=str(key))
    plt.legend()
    if show:
        plt.show()