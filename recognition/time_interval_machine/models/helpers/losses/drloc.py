import torch
import torch.nn.functional as F

def position_sampling(k, m, n):
    pos_1 = torch.randint(k, size=(n, m))
    pos_2 = torch.randint(k, size=(n, m))
    return pos_1, pos_2


def collect_samples(x, pos, n):
    _, l, D = x.size()
    x = x.permute(2,0,1).reshape(D, -1)
    pos = ((torch.arange(n).long().to(pos.device) * l).view(n,1) + pos).view(-1)
    return (x[:, pos]).view(D, n, -1).permute(1, 0, 2)


def dense_relative_localization_loss(x, model, m):
    n, l, D = x.size() # batch * length * feature_dim
    pos_1, pos_2 = position_sampling(l, m, n)

    deltax = torch.abs((pos_1 - pos_2).float()).to(x.device)
    deltax /= l

    pts_1 = collect_samples(x, pos_1, n).transpose(1, 2)
    pts_2 = collect_samples(x, pos_2, n).transpose(1, 2)
    predx = model(torch.cat([pts_1, pts_2], dim=2), "drloc_mlp")
    return F.l1_loss(deltax, predx)


def dense_relative_localization_loss_crossmodal(x1, x2, model, m):
    assert x1.size() == x2.size()
    n, l, D = x1.size() # batch * length * feature_dim
    pos_1, pos_2 = position_sampling(l, m, n)

    deltax = torch.abs((pos_1 - pos_2).float()).to(x1.device)
    deltax /= l

    pts_1 = collect_samples(x1, pos_1, n).transpose(1, 2)
    pts_2 = collect_samples(x2, pos_2, n).transpose(1, 2)
    predx = model(torch.cat([pts_1, pts_2], dim=2), "drloc_mlp")
    return F.l1_loss(deltax, predx)