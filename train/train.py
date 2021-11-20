import torch as t


def save_model(net, ep_num,
               name='weight',
               outPath='./weight/'):

    file_name = outPath + name + str(ep_num) + '.pth'
    t.save(net.state_dict(),
           file_name)
    print('Model Saved', file_name)


def load_model(file_path, model):
    state_dict = t.load(file_path)
    model.load_state_dict(state_dict)
    print('Model loaded', file_path)


def _train(net, trloader, loss_fn, opt,
           device = t.device('cuda' if t.cuda.is_available() else 'cpu'),
           ):
    batch_loss = list()
    for img, lbl in trloader:
        img = img.to(device= device)
        #lbl = lbl.to(device= device)

        pred = net( img)
        print('forward pass calculated')
        closs = loss_fn(pred, img, net)
        print("calced the loss")
        opt.zero_grad()
        closs.backward()
        opt.step()

        batch_loss.append(closs.item())
    return t.Tensor(batch_loss)

@t.no_grad()
def _validate(net, valoader, loss_fn,
           device = t.device('cuda' if t.cuda.is_available() else 'cpu'),
           ):
    batch_loss = list()
    for img, lbl in valoader:
        img = img.to(device=device)
        # lbl = lbl.to(device= device)
        pred = net(img)
        closs = loss_fn(pred, img, net)
        batch_loss.append(closs.item())
    return t.Tensor(batch_loss)

def train_validate(net, trloader, valoader,
                   epochs, loss_fn, opt,
                   device = t.device('cuda' if t.cuda.is_available() else 'cpu'),
                   ):
    tloss, vloss = list(), list()
    for e in range(epochs):
        tbloss = _train(net, trloader, loss_fn, opt, device)
        tloss.append(tloss)

        vbloss = _validate(net,valoader, loss_fn, device)
        vloss.append(vbloss)
        save_model(net, e)
        print("XXXXXXXXXXXXXXXXXXXXXX")
        print([ x.sum() for x in [tbloss, vbloss]])
        print('XXXXXXXXXXXXXXXXXXXXXX')
    return tloss, vloss



