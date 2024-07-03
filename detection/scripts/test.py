import pytorch_warmup as warmup
import numpy as np
import random
import wandb
import torch
import os

import time_interval_machine.datasets.loader as loader
import time_interval_machine.utils.logging as logging
import time_interval_machine.utils.distributed as du
import time_interval_machine.utils.checkpoint as ch

from time_interval_machine.models.helpers.losses.sigmoid import sigmoid_focal_loss
from time_interval_machine.models.helpers.losses.iou import ctr_diou_loss_1d
from time_interval_machine.models.helpers.losses.loss import get_loss
from time_interval_machine.utils.meters import InferenceMeter
from time_interval_machine.models.build import build_model

logger = logging.get_logger(__name__)

torch.backends.cudnn.benchmark = True

def init_test(args):
    assert args.pretrained_model != "", "No model specified in --pretrained_model"

    # Set up environment
    du.init_distributed_training(args)

    # Set random seeds
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    rng_generator = torch.manual_seed(args.seed)

    # Setup logging format.
    logging.setup_logging(args.output_dir)
    is_master_proc = du.is_master_proc(args.num_gpus * args.num_shards)

    logger.info("Output dir : {}".format(args.output_dir))
    
    criterion = sigmoid_focal_loss

    model, args = build_model(args)
    logger.info(model)

    ch.load_checkpoint(args, model)

    if "visual" in args.data_modality:
        if "train" in str(args.video_val_action_pickle):
            mode = "train"
        elif "validation" in str(args.video_val_action_pickle):
            mode = "val"
        else:
            mode = "test"
    else:
        if "train" in str(args.audio_val_action_pickle):
            mode = "train"
        elif "validation" in str(args.audio_val_action_pickle):
            mode = "val"
        else:
            mode = "test"

    val_loader = loader.create_loader(args, mode, args.data_modality, rng_generator)
    val_meter = InferenceMeter(args=args)
    _, _, _ = validate(
                    args=args,
                    val_loader=val_loader,
                    model=model,
                    criterion=criterion,
                    epoch=0,
                    is_master_proc=is_master_proc,
                    val_meter=val_meter,
                    wandb_log=False,
                    iters=0
                )

def validate(
            args,
            val_loader,
            model,
            criterion,
            epoch,
            is_master_proc,
            val_meter,
            normaliser=2000.0,
            wandb_log=False,
            iters=0
        ):
    with torch.no_grad():
        # Switch to evaluate mode
        model.eval()
        is_master_proc = du.is_master_proc(args.num_gpus * args.num_shards)
        val_meter.iter_tic()
        for i, (visual_input, audio_input, times, target, _) in enumerate(val_loader):
            # Put data onto GPU
            visual_input = visual_input.cuda(non_blocking=True)
            audio_input = audio_input.cuda(non_blocking=True)
            times = times.cuda(non_blocking=True)
            target = {k: v.cuda(non_blocking=True) for k, v in target.items()}

            # Measure data loading time
            val_meter.data_toc()

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

            # Compute output
            output, offsets, labels, _, ious = model(
                        [visual_input, audio_input],
                        "encoder",
                        times,
                        target,
                        label_queries=True
                    )
            val_meter.net_toc()

            # Visual side loss
            if ("visual" in args.data_modality):
                v_ious = ious[0]
                valid_reg_indices = (offsets[0][:, 0] != float("inf"))
                valid_cls_indices = (v_ious >= 0.0)
                num_pos = valid_reg_indices.sum()
                visual_targets = labels[0]
                v_ious = v_ious[valid_cls_indices]
                v_ious.masked_fill_((v_ious < args.iou_threshold), 1.0)

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

            # Gather losses onto 1 GPU
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

                loss = du.all_reduce([loss])[0]

            val_meter.update(
                    visual_loss.cpu().item(),
                    visual_loss_verb.cpu().item(),
                    visual_loss_noun.cpu().item(),
                    visual_loss_action.cpu().item(),
                    visual_reg_loss.cpu().item(),
                    audio_loss.cpu().item(),
                    audio_reg_loss.cpu().item(),
                    loss.cpu().item(),
                    positive_loss.cpu().item(),
                    negative_loss.cpu().item(),
                    visual_cls_num if isinstance(visual_cls_num, int) else visual_cls_num.cpu(),
                    audio_cls_num if isinstance(audio_cls_num, int) else audio_cls_num.cpu(),
                    visual_reg_num if isinstance(visual_reg_num, int) else visual_reg_num.cpu(),
                    audio_reg_num if isinstance(audio_reg_num, int) else audio_reg_num.cpu(),
                )

            # Measure elapsed time
            val_meter.iter_toc()
            if i % args.print_freq == 0 and is_master_proc:
                message = val_meter.get_val_message(
                        epoch,
                        i,
                        len(val_loader),
                    )
                logger.info(message)

            val_meter.iter_tic()

        best_loss, is_best = val_meter.update_epoch(epoch)

        # Log validation epoch stats
        if wandb_log and is_master_proc:
            iters += 1
            log_dict = val_meter.get_val_epoch_stats(iters)
            wandb.log(log_dict)
        logger.info(val_meter.get_val_epoch_message(epoch))

        val_meter.reset()

        return best_loss, is_best, iters