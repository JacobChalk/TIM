import torch
import numpy as np

def accuracy(output, target, topk=(1,5)):
    """
    Args:
        outputs: torch.FloatTensor, the tensor should be of shape
            [num_actions, class_count]
        labels: torch.LongTensor, the tensor should be of shape
            [num_actions]
        topk: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = max(topk)
    size = target.size(0)
    _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).to(torch.float32).sum(0)
        res.append(float(correct_k.mul_(100.0 / size)))
    return tuple(res)


def multitask_accuracy(outputs, labels, topk=(1,5)):
    """
    Args:
        outputs: tuple(torch.FloatTensor), each tensor should be of shape
            [num_actions, class_count]
        labels: tuple(torch.LongTensor), each tensor should be of shape
            [num_actions]
        topk: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(topk))
    task_count = len(outputs)
    size = labels[0].size(0)
    all_correct = torch.zeros(max_k, size).type(torch.ByteTensor)

    for output, label in zip(outputs, labels):
        _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct_for_task = pred.eq(label.view(1, -1).expand_as(pred))
        all_correct.add_(correct_for_task)
    accuracies = []
    for k in topk:
        all_tasks_correct = torch.ge(all_correct[:k].float().sum(0), task_count)
        accuracy_at_k = float(all_tasks_correct.float().sum(0) * 100.0 / size)
        accuracies.append(accuracy_at_k)
    return tuple(accuracies)
