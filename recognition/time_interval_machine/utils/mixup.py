import torch
import numpy as np

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda, inputs is the list of different data'''
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x[0].size(0)

    index = torch.randperm(batch_size).cuda()
    mixed_x = [lam * data + (1 - lam) * data[index, :] for data in x]

    if isinstance(y, dict):
        y_b = {
                'verb': y['verb'][index],
                'noun': y['noun'][index],
                'action': y['action'][index],
                'class_id': y['class_id'][index]
            }
    else:
        y_b = y[index]

    return mixed_x, y, y_b, lam

def mixup_criterion(criterion, pred_a, pred_b, y_a, y_b, lam, weights=None):
    loss_a = criterion(pred_a, y_a)
    if weights is not None:
        if isinstance(weights, (list, tuple)):
            loss_a = loss_a * weights[0]
        else:
            loss_a = loss_a * weights
    loss_a = loss_a.mean()
    loss_b = criterion(pred_b, y_b)
    if weights is not None:
        if isinstance(weights, (list, tuple)):
            loss_b = loss_b * weights[1]
        else:
            loss_b = loss_b * weights
    loss_b = loss_b.mean()
    return lam * loss_a + (1 - lam) * loss_b