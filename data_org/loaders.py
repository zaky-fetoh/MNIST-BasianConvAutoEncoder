import torchvision.datasets as dts
import torch.utils.data as data
import torchvision.transforms as trans



def getMNIST(train= True):
    mnist= dts.MNIST('./data_org/', download= True,
                     train=train, transform=trans.ToTensor())
    return mnist

def _getloaders(dataset, batch_size=10, shuffle=True,
               pin_memory= True, num_worker=2):
    return data.DataLoader(dataset,batch_size,shuffle,
                           num_workers=num_worker,
                           pin_memory=pin_memory)

def getloaders(batch_size=128, shuffle=True,
               pin_memory= True, num_worker=4):
    vdts, tdts = [getMNIST(i) for i in range(2)]
    return [_getloaders(dt) for dt in [vdts, tdts]]



