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

from time_interval_machine.utils.meters import TrainMeter, InferenceMeter
from time_interval_machine.utils.mixup import mixup_data, mixup_criterion
from time_interval_machine.utils.checkpoint import save_checkpoint
from time_interval_machine.models.build import build_model
from torch.nn import functional as F
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

    criterion = torch.nn.CrossEntropyLoss(
                                        label_smoothing=0.2,
                                        ignore_index=-1
                                    )

    training_iters = 0
    val_iters = 0
    if args.pretrained_model != "":
        start_epoch, checkpoint = ch.load_checkpoint(args, model)
    else:
        start_epoch = 0
        checkpoint = None

    train_loader = loader.create_loader(args, "train", args.data_modality, rng_generator)
    val_loader = loader.create_loader(args, "val", args.data_modality, rng_generator)

    train_meter = TrainMeter(args, train_loader.dataset.num_actions)
    val_meter = InferenceMeter(args, val_loader.dataset.num_actions)


    optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay
                        )
    num_steps = len(train_loader) * args.finetune_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                    optimizer,
                                                    T_max=num_steps,
                                                    eta_min=1e-6
                                                )
    warmup_scheduler = warmup.LinearWarmup(
                        optimizer,
                        warmup_period=len(train_loader) * args.warmup_epochs
                    )

    scaler = torch.cuda.amp.GradScaler()

    if checkpoint is not None and start_epoch != 0:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        warmup_scheduler.load_state_dict(checkpoint['warmup_scheduer'])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_meter.load_state_dict(checkpoint["train_meter"])
        val_meter.load_state_dict(checkpoint["val_meter"])
        scaler.load_state_dict(checkpoint["scaler"])
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

        training_iters = train_epoch(
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
                iters=training_iters
            )
        # Evaluate on validation set
        logger.info(f"Begin Audio-Visual Validation Epoch: [{epoch+1} / {args.finetune_epochs}]")
        best_acc1, is_best, stop, val_iters = validate(
                args=args,
                val_loader=val_loader,
                model=model,
                criterion=criterion,
                epoch=epoch,
                is_master_proc=is_master_proc,
                val_meter=val_meter,
                wandb_log=wandb_log,
                iters=val_iters
            )
        # Remember best acc@1 and save checkpoint
        if is_master_proc:
            sd = model.module.state_dict() if args.num_gpus > 1 else model.state_dict()
            save_checkpoint(
                    args,
                    {
                        'epoch': epoch + 1,
                        'state_dict': sd,
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'warmup_scheduer': warmup_scheduler.state_dict(),
                        'train_meter': train_meter.state_dict(),
                        'val_meter': val_meter.state_dict(),
                        'scaler': scaler.state_dict(),
                        'training_iters': training_iters,
                        'val_iters': val_iters
                    },
                    is_best
                )
        if stop:
            logger.info(f"Validation Accuracy has not improved after {args.early_stop_period+1} epochs. Stopping Training.")
            break

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
        iters=0
    ):

    # Switch to train mode
    model.train()

    train_meter.iter_tic()
    for i, (visual_input, audio_input, times, target, metadata) in enumerate(train_loader):
        # Put data onto GPU
        visual_input = visual_input.cuda(non_blocking=True)
        audio_input = audio_input.cuda(non_blocking=True)
        times = times.cuda(non_blocking=True)
        target = {k: v.cuda(non_blocking=True) for k, v in target.items()}

        metadata, v_queries, a_queries = misc.process_metadata(metadata)

        # Measure data loading time
        train_meter.data_toc()

        # Casts operations to mixed accision
        with torch.cuda.amp.autocast(enabled=args.enable_amp):
            time_encodings = model(times, "time_mlp")

            inputs = [visual_input, audio_input, time_encodings]
            inputs, target_a, target_b, lam = mixup_data(
                                                        inputs,
                                                        target,
                                                        alpha=args.mixup_alpha,
                                                    )
            time_encodings = inputs.pop()

            # Compute output
            output = model(
                        inputs,
                        "encoder",
                        time_encodings,
                        v_queries,
                        a_queries
                    )
            train_meter.net_toc()

            # For labels
            target_a = {k: torch.flatten(v) for k, v in target_a.items()}
            target_b = {k: torch.flatten(v) for k, v in target_b.items()}

            # Visual side loss
            valid_indices = (target_a['action'] != -1).cuda()
            valid_b_indices = (target_b['action'] != -1).cuda()

            valid_visual = valid_indices.sum()
            if ("visual" in args.data_modality) and (valid_visual > 0):
                    visual_target = {k: v[valid_indices] for k, v in target_a.items() if k != 'class_id'}
                    visual_b_target = {k: v[valid_b_indices] for k, v in target_b.items() if k != 'class_id'}

                    visual_target = torch.stack([v for v in visual_target.values()], dim=1)
                    visual_b_target = torch.stack([v for v in visual_b_target.values()], dim=1)

                    v_action_ids = metadata['v_action_ids'][valid_indices]

                    if args.include_verb_noun:
                        verb_preds = output[0][0][valid_indices]
                        noun_preds = output[0][1][valid_indices]

                        verb_b_preds = output[0][0][valid_b_indices]
                        noun_b_preds = output[0][1][valid_b_indices]

                        visual_loss_verb = mixup_criterion(
                                                        criterion,
                                                        verb_preds,
                                                        verb_b_preds,
                                                        visual_target[:, 0],
                                                        visual_b_target[:, 0],
                                                        lam
                                                    )
                        visual_loss_noun = mixup_criterion(
                                                        criterion,
                                                        noun_preds,
                                                        noun_b_preds,
                                                        visual_target[:, 1],
                                                        visual_b_target[:, 1],
                                                        lam
                                                    )
                    else:
                        visual_loss_verb = torch.FloatTensor([0.0]).detach()
                        visual_loss_noun = torch.FloatTensor([0.0]).detach()
                        verb_preds = torch.zeros(size=(visual_input.size(0), 97))
                        noun_preds = torch.zeros(size=(visual_input.size(0), 300))

                    action_preds = output[0][2][valid_indices]
                    action_b_preds = output[0][2][valid_b_indices]
                    visual_loss_action = torch.FloatTensor([0.0]).detach()
                    visual_loss_action = mixup_criterion(
                                        criterion,
                                        action_preds,
                                        action_b_preds,
                                        visual_target[:, 2],
                                        visual_b_target[:, 2],
                                        lam
                                    )

                    if args.include_verb_noun:
                        visual_loss = (visual_loss_verb + visual_loss_noun + visual_loss_action) / 3.0
                    else:
                        visual_loss = visual_loss_action

            else:
                visual_loss_verb = torch.FloatTensor([0.0]).detach()
                visual_loss_noun = torch.FloatTensor([0.0]).detach()
                visual_loss_action = torch.FloatTensor([0.0]).detach()
                visual_loss = torch.FloatTensor([0.0]).detach()

                verb_preds = torch.zeros(size=(visual_input.size(0), 97))
                noun_preds = torch.zeros(size=(visual_input.size(0), 300))
                action_preds = torch.zeros(size=(visual_input.size(0), 3806))
                v_action_ids = torch.empty(size=(visual_input.size(0),))
                visual_target = torch.empty(size=(visual_input.size(0), 3))


            # Audio side loss
            valid_indices = (target_a['class_id'] != -1).cuda()
            valid_b_indices = (target_b['class_id'] != -1).cuda()

            valid_audio = valid_indices.sum()
            if ("audio" in args.data_modality) and (valid_audio > 0):
                audio_preds = output[0][3][valid_indices]
                audio_b_preds = output[0][3][valid_b_indices]

                audio_target = target_a['class_id'][valid_indices]
                audio_b_target = target_b['class_id'][valid_b_indices]

                audio_loss = mixup_criterion(
                                        criterion,
                                        audio_preds,
                                        audio_b_preds,
                                        audio_target,
                                        audio_b_target,
                                        lam
                                    )

                a_action_ids = metadata['a_action_ids'][valid_indices]
            else:
                audio_loss = torch.FloatTensor([0.0]).detach()
                audio_preds = torch.zeros(size=(audio_input.size(0), 44))
                audio_target = torch.empty(size=(audio_input.size(0),))
                a_action_ids = torch.empty(size=(audio_input.size(0),))

            loss = torch.FloatTensor([0.0]).cuda()
            if args.data_modality == "visual":
                loss += visual_loss
            elif args.data_modality == "audio":
                loss += audio_loss
            else:
                loss += (visual_loss + args.lambda_audio * audio_loss)

            # Dense relative localization loss
            if args.lambda_drloc > 0.0:
                if args.model_modality == "audio_visual":
                    length = args.num_feats
                    drloc_loss = drl.dense_relative_localization_loss_crossmodal(
                                        output[1][:, :length],
                                        output[1][:, length:],
                                        model,
                                        args.m_drloc
                                    )
                else:
                    drloc_loss = drl.dense_relative_localization_loss(
                                        output[1],
                                        model,
                                        args.m_drloc
                                    )
                loss += (args.lambda_drloc * drloc_loss)
            else:
                drloc_loss = torch.FloatTensor([0.0]).detach()

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
            verb_preds = du.all_gather([verb_preds])
            noun_preds = du.all_gather([noun_preds])
            action_preds = du.all_gather([action_preds])
            audio_preds = du.all_gather([audio_preds])
            v_action_ids = du.all_gather([v_action_ids])
            a_action_ids = du.all_gather([a_action_ids])
            visual_target = du.all_gather([visual_target])
            audio_target = du.all_gather([audio_target])
            visual_loss = du.all_reduce([visual_loss])[0]
            visual_loss_verb = du.all_reduce([visual_loss_verb])[0]
            visual_loss_noun = du.all_reduce([visual_loss_noun])[0]
            visual_loss_action = du.all_reduce([visual_loss_action])[0]
            audio_loss = du.all_reduce([audio_loss])[0]
            loss = du.all_reduce([loss])[0]
            drloc_loss = du.all_reduce([drloc_loss])[0]
            valid_visual = du.all_gather([torch.IntTensor(valid_visual)]).sum()
            valid_audio = du.all_gather([torch.IntTensor(valid_audio)]).sum()

        # Track losses
        train_meter.update(
            verb_preds.detach().cpu(),
            noun_preds.detach().cpu(),
            action_preds.detach().cpu(),
            audio_preds.detach().cpu(),
            v_action_ids.detach().cpu(),
            a_action_ids.detach().cpu(),
            visual_target.detach().cpu(),
            audio_target.detach().cpu(),
            visual_loss.item(),
            visual_loss_verb.item(),
            visual_loss_noun.item(),
            visual_loss_action.item(),
            audio_loss.item(),
            loss.item(),
            drloc_loss.item(),
            valid_visual,
            valid_audio
        )

        # Measure elapsed time
        train_meter.iter_toc()

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
                                            iters
                                        )
                wandb.log(log_dict)
            logger.info(message)

        train_meter.iter_tic()

    train_meter.update_epoch()

    # Log train epoch stats
    if wandb_log and is_master_proc:
        iters += 1
        log_dict = train_meter.get_train_epoch_stats(iters)
        wandb.log(log_dict)
    logger.info(train_meter.get_train_epoch_message(epoch))

    train_meter.reset()

    return iters
