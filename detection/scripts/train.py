import pytorch_warmup as warmup
import numpy as np
import random
import wandb
import torch
import os

import time_interval_machine.models.helpers.losses.drloc as drl
import time_interval_machine.datasets.loader as loader
import time_interval_machine.utils.logging as logging
import time_interval_machine.utils.distributed as du
import time_interval_machine.utils.checkpoint as ch
import time_interval_machine.utils.misc as misc

from time_interval_machine.models.helpers.losses.sigmoid import sigmoid_focal_loss
from time_interval_machine.models.helpers.losses.iou import ctr_diou_loss_1d
from time_interval_machine.utils.meters import TrainMeter, InferenceMeter
from time_interval_machine.models.helpers.losses.loss import get_loss
from time_interval_machine.utils.checkpoint import save_checkpoint
from time_interval_machine.models.build import build_model
from scripts.test import validate

logger = logging.get_logger(__name__)

torch.backends.cudnn.benchmark = True

def init_train(args):
    # Set up environment
    du.init_distributed_training(args)

    # Set random seeds
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    rng_generator = torch.manual_seed(args.seed)

    # Setup logging format.
    logging.setup_logging(args.output_dir)
    logger.info(f"Random Seed is: {args.seed}")
    is_master_proc = du.is_master_proc(args.num_gpus * args.num_shards)

    model, args = build_model(args)
    logger.info(model)

    logger.info("Output dir : {}".format(args.output_dir))

    criterion = sigmoid_focal_loss

    training_iters = 0
    val_iters = 0
    if args.pretrained_model != "":
        start_epoch, checkpoint = ch.load_checkpoint(args, model)
    else:
        start_epoch = 0
        checkpoint = None

    train_loader = loader.create_loader(args, "train", args.data_modality, rng_generator)
    val_loader = loader.create_loader(args, "val", args.data_modality, rng_generator)

    train_meter = TrainMeter(args)
    val_meter = InferenceMeter(args)

    optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay
                        )
    num_steps = len(train_loader) * args.finetune_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                    optimizer,
                                                    T_max=num_steps,
                                                    eta_min=1e-5
                                                )
    warmup_scheduler = warmup.LinearWarmup(
                        optimizer,
                        warmup_period=len(train_loader) * args.warmup_epochs
                    )

    scaler = torch.cuda.amp.GradScaler()

    normaliser = args.normaliser
    if checkpoint is not None and start_epoch != 0:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        warmup_scheduler.load_state_dict(checkpoint['warmup_scheduer'])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_meter.load_state_dict(checkpoint["train_meter"])
        val_meter.load_state_dict(checkpoint["val_meter"])
        scaler.load_state_dict(checkpoint["scaler"])
        normaliser = checkpoint["normaliser"]
        training_iters = checkpoint["training_iters"]
        val_iters = checkpoint["val_iters"]

    if args.enable_wandb_log and is_master_proc:
        wandb_log = True
        wandb.init(project='TIM', config=args, mode="offline")
        wandb.run.log_code(".")
        wandb.watch(model)
    else:
        wandb_log = False

    for epoch in range(start_epoch, args.finetune_epochs):
        logger.info(f"Begin Audio-Visual Train Epoch: [{epoch+1} / {args.finetune_epochs}]")

        if args.num_gpus > 1:
            train_loader.sampler.set_epoch(epoch)

        training_iters, normaliser = train_epoch(
                args,
                train_loader,
                model,
                criterion,
                optimizer,
                lr_scheduler,
                warmup_scheduler,
                train_meter,
                scaler,
                epoch,
                is_master_proc,
                wandb_log=wandb_log,
                iters=training_iters,
                normaliser=normaliser
            )
        
        logger.info(f"Begin Audio-Visual Validation Epoch: [{epoch+1} / {args.finetune_epochs}]")
        best_loss, is_best, val_iters = validate(
                args=args,
                val_loader=val_loader,
                model=model,
                criterion=criterion,
                epoch=epoch,
                is_master_proc=is_master_proc,
                val_meter=val_meter,
                wandb_log=wandb_log,
                iters=val_iters,
                normaliser=normaliser
            )
        # Save checkpoint
        if is_master_proc:
            sd = model.module.state_dict() if args.num_gpus > 1 else model.state_dict()
            save_checkpoint(
                    args,
                    {
                        'epoch': epoch + 1,
                        'state_dict': sd,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'warmup_scheduer': warmup_scheduler.state_dict(),
                        'train_meter': train_meter.state_dict(),
                        'val_meter': val_meter.state_dict(),
                        'scaler': scaler.state_dict(),
                        'normaliser': normaliser,
                        'training_iters': training_iters,
                        'val_iters': val_iters
                    },
                    is_best
                )
    if is_master_proc:
        wandb.finish()

def train_epoch(
        args,
        train_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        warmup_scheduler,
        train_meter,
        scaler,
        epoch,
        is_master_proc,
        wandb_log=False,
        iters=0,
        normaliser=250.0
    ):

    # Switch to train mode
    model.train()

    train_meter.iter_tic()
    for i, (visual_input, audio_input, times, target, _) in enumerate(train_loader):
        # Put data onto GPU
        visual_input = visual_input.cuda(non_blocking=True)
        audio_input = audio_input.cuda(non_blocking=True)
        times = times.cuda(non_blocking=True)
        target = {k: v.cuda(non_blocking=True) for k, v in target.items()}

        # Measure data loading time
        train_meter.data_toc()

        # Casts operations to mixed accision
        with torch.cuda.amp.autocast(enabled=args.enable_amp):

            visual_loss_verb = torch.FloatTensor([0.0]).detach().cuda()
            visual_loss_noun = torch.FloatTensor([0.0]).detach().cuda()
            visual_loss_action = torch.FloatTensor([0.0]).detach().cuda()
            visual_loss = torch.FloatTensor([0.0]).detach().cuda()
            visual_reg_loss = torch.FloatTensor([0.0]).detach().cuda()
            visual_cls_num = 0
            visual_reg_num = 0

            audio_loss = torch.FloatTensor([0.0]).detach().cuda()
            audio_reg_loss = torch.FloatTensor([0.0]).detach().cuda()
            audio_cls_num = 0
            audio_reg_num = 0

            loss = torch.FloatTensor([0.0]).cuda()
            drloc_loss = torch.FloatTensor([0.0]).detach()

            # Compute output
            output, offsets, labels, _, ious = model(
                        [visual_input, audio_input],
                        "encoder",
                        times,
                        target,
                        label_queries=True
                    )
            train_meter.net_toc()


            # Visual side loss
            if ("visual" in args.data_modality):
                    v_ious = ious[0]
                    valid_reg_indices = (offsets[0][:, 0] != float("inf"))
                    valid_cls_indices = (v_ious >= 0.0)
                    num_pos = valid_reg_indices.sum()
                    visual_targets = labels[0]
                    v_ious = v_ious[valid_cls_indices]
                    v_ious.masked_fill_((v_ious < args.iou_threshold), 1.0)

                    normaliser = (args.normaliser_momentum * normaliser) + ((1.0 - args.normaliser_momentum) * max(num_pos, 1))
                    if args.include_verb_noun:
                        verb_preds = output[0][0][valid_cls_indices]
                        noun_preds = output[0][1][valid_cls_indices]

                        visual_loss_verb = get_loss(
                                                criterion,
                                                verb_preds,
                                                visual_targets[0][valid_cls_indices],
                                                weights=v_ious,
                                                reduction="sum"
                                            )
                        visual_loss_noun = get_loss(
                                                criterion,
                                                noun_preds,
                                                visual_targets[1][valid_cls_indices],
                                                weights=v_ious,
                                                reduction="sum"
                                            )

                    action_preds = output[0][2][valid_cls_indices]

                    visual_loss_action = get_loss(
                                        criterion,
                                        action_preds,
                                        visual_targets[2][valid_cls_indices],
                                        weights=v_ious,
                                        reduction="sum"
                                    )

                    with torch.no_grad():
                        positive_inds = valid_reg_indices
                        negative_inds = (~positive_inds)
                        split_loss = get_loss(
                                            criterion,
                                            action_preds,
                                            visual_targets[2][valid_cls_indices],
                                            weights=v_ious,
                                            reduction="none"
                                        )
                        positive_loss = (split_loss[positive_inds].sum()) / normaliser
                        negative_loss = (split_loss[negative_inds].sum()) / normaliser

                    if args.include_verb_noun:
                        visual_loss = (visual_loss_verb + visual_loss_noun + visual_loss_action) / (3.0 * normaliser)
                    else:
                        visual_loss = visual_loss_action / normaliser

                    if num_pos > 0:
                        action_reg = output[1][0][valid_reg_indices]
                        reg_loss_action = get_loss(
                                                    ctr_diou_loss_1d,
                                                    action_reg,
                                                    offsets[0][valid_reg_indices],
                                                    reduction="sum"
                                                ) * args.lambda_reg

                        visual_reg_loss = reg_loss_action / normaliser

                    visual_cls_num = valid_cls_indices.sum()
                    visual_reg_num = num_pos

            # Audio side loss
            if ("audio" in args.data_modality):
                a_ious = ious[1]
                valid_reg_indices = (offsets[1][:, 0] != float("inf"))
                valid_cls_indices = (a_ious >= 0.0)
                num_pos = valid_reg_indices.sum()
                audio_targets = labels[1]
                a_ious = a_ious[valid_cls_indices]
                a_ious.masked_fill_((a_ious < args.iou_threshold), 1.0)

                normaliser = (args.normaliser_momentum * normaliser) + ((1.0 - args.normaliser_momentum) * max(num_pos, 1))
                audio_preds = output[0][3][valid_cls_indices]
                audio_loss = get_loss(
                                        sigmoid_focal_loss,
                                        audio_preds,
                                        audio_targets[valid_cls_indices],
                                        weights=a_ious,
                                        reduction="sum"
                                    ) / normaliser


                with torch.no_grad():
                    positive_inds = valid_reg_indices
                    negative_inds = (~positive_inds)
                    split_loss = get_loss(
                                        criterion,
                                        audio_preds,
                                        audio_targets[valid_cls_indices],
                                        weights=a_ious,
                                        reduction="none"
                                    )
                    positive_loss = (split_loss[positive_inds].sum()) / normaliser
                    negative_loss = (split_loss[negative_inds].sum()) / normaliser


                if num_pos > 0:
                    audio_reg = output[1][1][valid_reg_indices]

                    audio_reg_loss = get_loss(
                                            ctr_diou_loss_1d,
                                            audio_reg,
                                            offsets[1][valid_reg_indices],
                                            reduction="sum"
                                        ) * args.lambda_reg
                    audio_reg_loss = audio_reg_loss / normaliser

                audio_cls_num = valid_cls_indices.sum()
                audio_reg_num = num_pos

            if args.data_modality == "visual":
                loss += visual_loss + visual_reg_loss
            elif args.data_modality == "audio":
                loss += audio_loss + audio_reg_loss
            else:
                loss += (visual_loss + visual_reg_loss)
                loss += args.lambda_audio * (audio_loss + audio_reg_loss)

            # Dense relative localization loss
            if args.lambda_drloc > 0.0:
                if args.model_modality == "audio_visual":
                    length = args.num_feats
                    drloc_loss = drl.dense_relative_localization_loss_crossmodal(
                                        output[2][:, :length],
                                        output[2][:, length:],
                                        model,
                                        args.m_drloc
                                    )
                else:
                    drloc_loss = drl.dense_relative_localization_loss(
                                        output[2],
                                        model,
                                        args.m_drloc
                                    )
                loss += (args.lambda_drloc * drloc_loss)

        misc.check_nan_losses(loss)

        # Compute gradient and backprop
        optimizer.zero_grad()
        if args.enable_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with warmup_scheduler.dampening():
            lr_scheduler.step()

        # Collect losses onto one GPU before update
        if args.num_gpus > 1:
            if "visual" in args.data_modality:
                if args.include_verb_noun:
                    visual_loss_verb = du.all_reduce([visual_loss_verb])[0]
                    visual_loss_noun = du.all_reduce([visual_loss_noun])[0]
                action_reg = du.all_gather([action_reg])

                visual_loss = du.all_reduce([visual_loss])[0]
                visual_loss_action = du.all_reduce([visual_loss_action])[0]
                visual_reg_loss = du.all_reduce([visual_reg_loss])[0]

            if "audio" in args.data_modality:
                audio_loss = du.all_reduce([audio_loss])[0]
                audio_reg_loss = du.all_reduce([audio_reg_loss])[0]

            if args.lambda_drloc > 0.0:
                drloc_loss = du.all_reduce([drloc_loss])[0]

            loss = du.all_reduce([loss])[0]

        # Measure elapsed time
        train_meter.iter_toc()

        # Track losses
        train_meter.update(
            visual_loss.detach().cpu().item(),
            (visual_loss_verb / normaliser).detach().cpu().item(),
            (visual_loss_noun / normaliser).detach().cpu().item(),
            (visual_loss_action / normaliser).detach().cpu().item(),
            visual_reg_loss.detach().cpu().item(),
            audio_loss.detach().cpu().item(),
            audio_reg_loss.detach().cpu().item(),
            loss.detach().cpu().item(),
            drloc_loss.detach().cpu().item(),
            positive_loss.detach().cpu().item(),
            negative_loss.detach().cpu().item(),
            visual_cls_num if isinstance(visual_cls_num, int) else visual_cls_num.cpu(),
            audio_cls_num if isinstance(audio_cls_num, int) else audio_cls_num.cpu(),
            visual_reg_num if isinstance(visual_reg_num, int) else visual_reg_num.cpu(),
            audio_reg_num if isinstance(audio_reg_num, int) else audio_reg_num.cpu(),
        )

        if i % args.print_freq == 0 and is_master_proc:
            message = train_meter.get_train_message(
                    epoch,
                    i,
                    len(train_loader),
                    optimizer.param_groups[-1]['lr']
                )
            if wandb_log:
                iters += 1
                log_dict = train_meter.get_train_stats(
                                            optimizer.param_groups[-1]['lr'],
                                            iters,
                                            normaliser,
                                            num_pos,
                                            (visual_cls_num - num_pos) if 'visual' in args.data_modality else (audio_cls_num - num_pos)
                                        )
                wandb.log(log_dict)
            logger.info(message)

        train_meter.iter_tic()

    # Log validation epoch stats
    if wandb_log and is_master_proc:
        iters += 1
        log_dict = train_meter.get_train_epoch_stats(iters)
        wandb.log(log_dict)
    logger.info(train_meter.get_train_epoch_message(epoch))

    train_meter.reset()

    return iters, normaliser
