# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import os
import sys
from typing import Iterable, Optional

import numpy as np
import torch
from scipy.special import softmax
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import utils


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    if isinstance(target, tuple):
        loss_v = criterion(outputs[0], target[0])
        loss_n = criterion(outputs[1], target[1])
        loss = (loss_v, loss_n)
    else:
        loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(
        optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    args,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    log_writer=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=None,
                    update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group[
                        "lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        if isinstance(targets, list):
            targets_v = targets[0].to(device, non_blocking=True)
            targets_n = targets[1].to(device, non_blocking=True)
            targets = (targets_v, targets_n)
        else:
            targets = targets.to(device, non_blocking=True)

        if args.num_segment > 1:
            samples = samples.view((-1, ) + samples.size()[2:])

        if mixup_fn is not None:
            B, C, T, H, W = samples.shape
            samples = samples.view(B, C * T, H, W)
            samples, targets = mixup_fn(samples, targets)
            samples = samples.view(B, C, T, H, W)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(model, samples, targets,
                                             criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(model, samples, targets,
                                                 criterion)
        if isinstance(loss, tuple):
            loss_v = loss[0]
            loss_n = loss[1]
            loss = loss_v + loss_n

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            grad_norm = model.get_global_grad_norm()

            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss,
                                    optimizer,
                                    clip_grad=max_norm,
                                    parameters=model.parameters(),
                                    create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) %
                                    update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            if isinstance(targets, tuple):
                verb_acc = (output[0].max(-1)[-1] == targets[0]).float().mean()
                noun_acc = (output[1].max(-1)[-1] == targets[1]).float().mean()
            else:
                class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            if isinstance(targets, tuple):
                verb_acc = None
                noun_acc = None
            else:
                class_acc = None

        metric_logger.update(loss=loss_value)
        if isinstance(targets, tuple):
            metric_logger.update(verb_loss=loss_v.item())
            metric_logger.update(noun_loss=loss_n.item())
            metric_logger.update(verb_acc=verb_acc)
            metric_logger.update(noun_acc=noun_acc)
        else:
            metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if isinstance(targets, tuple):
                log_writer.update(verb_loss=loss_v.item(), head="loss")
                log_writer.update(noun_loss=loss_n.item(), head="loss")
                log_writer.update(verb_acc=verb_acc, head="loss")
                log_writer.update(noun_acc=noun_acc, head="loss")
            else:
                log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        if isinstance(target, list):
            target_v = target[0].to(device, non_blocking=True)
            target_n = target[1].to(device, non_blocking=True)
            target = (target_v, target_n)
        else:
            target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if isinstance(target, tuple):
                loss_v = criterion(output[0], target[0])
                loss_n = criterion(output[1], target[1])
                loss = loss_v + loss_n
            else:
                loss = criterion(output, target)

        if isinstance(target, tuple):
            verb_acc1, verb_acc5 = accuracy(output[0], target[0], topk=(1, 5))
            noun_acc1, noun_acc5 = accuracy(output[1], target[1], topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        if isinstance(target, tuple):
            metric_logger.update(verb_loss=loss_v.item())
            metric_logger.update(noun_loss=loss_n.item())
            metric_logger.meters['verb_acc1'].update(verb_acc1.item(), n=batch_size)
            metric_logger.meters['verb_acc5'].update(verb_acc5.item(), n=batch_size)
            metric_logger.meters['noun_acc1'].update(noun_acc1.item(), n=batch_size)
            metric_logger.meters['noun_acc5'].update(noun_acc5.item(), n=batch_size)
        else:
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if isinstance(target, tuple):
        print(
            '* Verb_Acc@1 {verb_top1.global_avg:.3f} Verb_Acc@5 {verb_top5.global_avg:.3f} Noun_Acc@1 {noun_top1.global_avg:.3f} Noun_Acc@5 {noun_top5.global_avg:.3f} loss {losses.global_avg:.3f} Verb loss {verb_losses.global_avg:.3f} Noun loss {noun_losses.global_avg:.3f}'
            .format(verb_top1=metric_logger.verb_acc1,
                    verb_top5=metric_logger.verb_acc5,
                    noun_top1=metric_logger.noun_acc1,
                    noun_top5=metric_logger.noun_acc5,
                    losses=metric_logger.loss,
                    verb_losses=metric_logger.verb_loss,
                    noun_losses=metric_logger.noun_loss))
    else:
        print(
            '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1,
                    top5=metric_logger.acc5,
                    losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        images = images.to(device, non_blocking=True)
        if isinstance(target, list):
            target_v = target[0].to(device, non_blocking=True)
            target_n = target[1].to(device, non_blocking=True)
            target = (target_v, target_n)
        else:
            target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if isinstance(target, tuple):
                loss_v = criterion(output[0], target[0])
                loss_n = criterion(output[1], target[1])
                loss = loss_v +loss_n
            else:
                loss = criterion(output, target)

        if isinstance(target, tuple):
            for i in range(output.size(0)):
                string = "{} {} {} {} {} {} {}\n".format(ids[i], \
                                                         str(output[0].data[i].cpu().numpy().tolist()), \
                                                         str(output[1].data[i].cpu().numpy().tolist()), \
                                                         str(int(target[0][i].cpu().numpy())), \
                                                         str(int(target[1][i].cpu().numpy())), \
                                                         str(int(chunk_nb[i].cpu().numpy())), \
                                                         str(int(split_nb[i].cpu().numpy())))
                final_result.append(string)
        else:
            for i in range(output.size(0)):
                string = "{} {} {} {} {}\n".format(ids[i], \
                                                    str(output.data[i].cpu().numpy().tolist()), \
                                                    str(int(target[i].cpu().numpy())), \
                                                    str(int(chunk_nb[i].cpu().numpy())), \
                                                    str(int(split_nb[i].cpu().numpy())))
                final_result.append(string)

        if isinstance(target, tuple):
            verb_acc1, verb_acc5 = accuracy(output[0], target[0], topk=(1, 5))
            noun_acc1, noun_acc5 = accuracy(output[1], target[1], topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        if isinstance(target, tuple):
            metric_logger.update(verb_loss=loss_v.item())
            metric_logger.update(noun_loss=loss_n.item())
            metric_logger.meters['verb_acc1'].update(verb_acc1.item(), n=batch_size)
            metric_logger.meters['verb_acc5'].update(verb_acc5.item(), n=batch_size)
            metric_logger.meters['noun_acc1'].update(noun_acc1.item(), n=batch_size)
            metric_logger.meters['noun_acc5'].update(noun_acc5.item(), n=batch_size)
        else:
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        if isinstance(target, tuple):
            f.write("{}, {}, {}, {}\n".format(verb_acc1, verb_acc5, noun_acc1, noun_acc5))
        else:
            f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if isinstance(target, tuple):
        print(
            '* Verb_Acc@1 {verb_top1.global_avg:.3f} Verb_Acc@5 {verb_top5.global_avg:.3f} Noun_Acc@1 {noun_top1.global_avg:.3f} Noun_Acc@5 {noun_top5.global_avg:.3f} loss {losses.global_avg:.3f} Verb loss {verb_losses.global_avg:.3f} Noun loss {noun_losses.global_avg:.3f}'
            .format(verb_top1=metric_logger.verb_acc1,
                    verb_top5=metric_logger.verb_acc5,
                    noun_top1=metric_logger.noun_acc1,
                    noun_top5=metric_logger.noun_acc5,
                    losses=metric_logger.loss,
                    verb_losses=metric_logger.verb_loss,
                    noun_losses=metric_logger.noun_loss))
    else:
        print(
            '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1,
                    top5=metric_logger.acc5,
                    losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, method='prob', multitask=False):
    assert method in ['prob', 'score']
    if multitask:
        dict_feats_v = {}
        dict_feats_n = {}
        dict_label_v = {}
        dict_label_n = {}
    else:
        dict_feats = {}
        dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            if multitask:
                label_v = line.split(']')[2].split(' ')[1]
                label_n = line.split(']')[2].split(' ')[2]
                chunk_nb = line.split(']')[2].split(' ')[3]
                split_nb = line.split(']')[2].split(' ')[4]
                data_v = np.fromstring(line.split('[')[1].split(']')[0],
                                    dtype=np.float,
                                    sep=',')
                data_n = np.fromstring(line.split('[')[2].split(']')[0],
                                    dtype=np.float,
                                    sep=',')
                if not name in dict_feats_v:
                    dict_feats_v[name] = []
                    dict_feats_n[name] = []
                    dict_label_v[name] = 0
                    dict_label_n[name] = 0
                    dict_pos[name] = []
            else:
                label = line.split(']')[1].split(' ')[1]
                chunk_nb = line.split(']')[1].split(' ')[2]
                split_nb = line.split(']')[1].split(' ')[3]
                data = np.fromstring(line.split('[')[1].split(']')[0],
                                    dtype=np.float,
                                    sep=',')
                if not name in dict_feats:
                    dict_feats[name] = []
                    dict_label[name] = 0
                    dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            if method == 'prob':
                if multitask:
                    dict_feats_v[name].append(softmax(data_v))
                    dict_feats_n[name].append(softmax(data_n))
                else:
                    dict_feats[name].append(softmax(data))
            else:
                if multitask:
                    dict_feats_v[name].append(data_v)
                    dict_feats_n[name].append(data_n)
                else:
                    dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            if multitask:
                dict_label_v[name] = label_v
                dict_label_n[name] = label_n
            else:
                dict_label[name] = label
    print("Computing final results")

    if multitask:
        input_lst_v = []
        print(len(dict_feats_v))
        for i, item in enumerate(dict_feats_v):
            input_lst_v.append([i, item, dict_feats_v[item], dict_label_v[item]])
        from multiprocessing import Pool
        p = Pool(64)
        ans_v = p.map(compute_video, input_lst_v)
        top1_v = [x[1] for x in ans_v]
        top5_v = [x[2] for x in ans_v]
        pred_v = [x[0] for x in ans_v]
        label_v = [x[3] for x in ans_v]
        final_top1_v, final_top5_v = np.mean(top1_v), np.mean(top5_v)

        input_lst_n = []
        print(len(dict_feats_n))
        for i, item in enumerate(dict_feats_n):
            input_lst_n.append([i, item, dict_feats_n[item], dict_label_n[item]])
        from multiprocessing import Pool
        p = Pool(64)
        ans_n = p.map(compute_video, input_lst_n)
        top1_n = [x[1] for x in ans_n]
        top5_n = [x[2] for x in ans_n]
        pred_n = [x[0] for x in ans_n]
        label_n = [x[3] for x in ans_n]
        final_top1_n, final_top5_n = np.mean(top1_n), np.mean(top5_n)

        # print(final_top1*100 ,final_top5*100)
        return final_top1_v * 100, final_top5_v * 100, final_top1_n * 100, final_top5_n * 100
    else:
        input_lst = []
        print(len(dict_feats))
        for i, item in enumerate(dict_feats):
            input_lst.append([i, item, dict_feats[item], dict_label[item]])
        from multiprocessing import Pool
        p = Pool(64)
        ans = p.map(compute_video, input_lst)
        top1 = [x[1] for x in ans]
        top5 = [x[2] for x in ans]
        pred = [x[0] for x in ans]
        label = [x[3] for x in ans]
        final_top1, final_top5 = np.mean(top1), np.mean(top5)

        # print(final_top1*100 ,final_top5*100)
        return final_top1 * 100, final_top5 * 100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
