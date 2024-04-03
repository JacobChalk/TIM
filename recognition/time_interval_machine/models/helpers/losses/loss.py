import time_interval_machine.utils.logging as logging

logger = logging.get_logger(__name__)

def get_loss(criterion, pred, y, weights=None, reduction='mean'):
    loss = criterion(pred, y)
    if weights is not None:
        loss = loss * weights[:, None]
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss