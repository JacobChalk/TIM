import numpy as np
import random
import wandb
import torch
import os

import time_interval_machine.datasets.loader as loader
import time_interval_machine.utils.logging as logging
import time_interval_machine.utils.distributed as du
import time_interval_machine.utils.checkpoint as ch
import time_interval_machine.utils.misc as misc

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

    criterion = torch.nn.CrossEntropyLoss(
                                        label_smoothing=0.2,
                                        ignore_index=-1
                                    )

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
    val_meter = InferenceMeter(args, val_loader.dataset.num_actions)
    _, _, _, _ = validate(
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
            wandb_log=False,
            iters=0
        ):
    with torch.no_grad():
        # Switch to evaluate mode
        model.eval()
        is_master_proc = du.is_master_proc(args.num_gpus * args.num_shards)
        val_meter.iter_tic()
        for i, (visual_input, audio_input, times, target, metadata) in enumerate(val_loader):
            # Put data onto GPU
            visual_input = visual_input.cuda(non_blocking=True)
            audio_input = audio_input.cuda(non_blocking=True)
            times = times.cuda(non_blocking=True)
            target = {k: v.cuda(non_blocking=True) for k, v in target.items()}

            metadata, v_queries, a_queries = misc.process_metadata(metadata)

            target = {k: torch.flatten(v) for k, v in target.items()}

            # Measure data loading time
            val_meter.data_toc()

            inputs = [visual_input, audio_input]
            time_encodings = model(times, "time_mlp")

            # Compute output
            output = model(
                            inputs,
                            "encoder",
                            time_encodings,
                            v_queries,
                            a_queries
                        )
            val_meter.net_toc()


            # Compute visual loss
            valid_indices = (target['action'] != -1).cuda()
            valid_visual = valid_indices.sum()
            if ("visual" in args.data_modality) and (valid_visual > 0):
                v_target = {k: v[valid_indices] for k, v in target.items() if k != 'class_id'}
                v_target = torch.stack([v for v in v_target.values()], dim=1)
                v_action_ids = metadata['v_action_ids'][valid_indices]

                if args.include_verb_noun:
                    verb_preds = output[0][0][valid_indices]
                    noun_preds = output[0][1][valid_indices]
                    visual_loss_verb = criterion(verb_preds, v_target[:, 0])
                    visual_loss_noun = criterion(noun_preds, v_target[:, 1])
                else:
                    visual_loss_verb = torch.FloatTensor([0.0])
                    visual_loss_noun = torch.FloatTensor([0.0])
                    verb_preds = torch.zeros(size=(visual_input.size(0), 97)).detach()
                    noun_preds = torch.zeros(size=(visual_input.size(0), 300)).detach()

                action_preds = output[0][2][valid_indices]
                visual_loss_action = criterion(action_preds, v_target[:, 2])

                if args.include_verb_noun:
                    visual_loss = (visual_loss_verb + visual_loss_noun + visual_loss_action) / 3.0
                else:
                    visual_loss = visual_loss_action

            else:
                visual_loss_verb = torch.FloatTensor([0.0])
                visual_loss_noun = torch.FloatTensor([0.0])
                visual_loss_action = torch.FloatTensor([0.0])
                visual_loss = torch.FloatTensor([0.0])

                verb_preds = torch.zeros(size=(visual_input.size(0), 97)).detach()
                noun_preds = torch.zeros(size=(visual_input.size(0), 300)).detach()
                action_preds = torch.zeros(size=(visual_input.size(0), 3806)).detach()
                v_target = torch.empty(size=(visual_input.size(0), 3))
                v_action_ids = torch.empty(size=(visual_input.size(0),))

            # Compute audio loss
            valid_indices = (target['class_id'] != -1).cuda()
            valid_audio = valid_indices.sum()
            if ("audio" in args.data_modality) and (valid_audio > 0):
                aud_preds = output[0][3][valid_indices]

                a_target = target['class_id'][valid_indices]

                audio_loss = criterion(aud_preds, a_target)

                a_action_ids = metadata['a_action_ids'][valid_indices]
            else:
                audio_loss = torch.FloatTensor([0.0])
                aud_preds = torch.zeros(size=(audio_input.size(0), 3806)).detach()
                a_target = torch.empty(size=(audio_input.size(0),))
                a_action_ids = torch.empty(size=(audio_input.size(0),))

            # Gather losses onto 1 GPU
            if args.num_gpus > 1:
                verb_preds = du.all_gather([verb_preds])
                noun_preds = du.all_gather([noun_preds])
                action_preds = du.all_gather([action_preds])
                aud_preds = du.all_gather([aud_preds])
                v_action_ids = du.all_gather([v_action_ids])
                a_action_ids = du.all_gather([a_action_ids])
                v_target = du.all_gather([v_target])
                a_target = du.all_gather([a_target])
                visual_loss = du.all_reduce([visual_loss])[0]
                visual_loss_verb = du.all_reduce([visual_loss_verb])[0]
                visual_loss_noun = du.all_reduce([visual_loss_noun])[0]
                visual_loss_action = du.all_reduce([visual_loss_action])[0]
                audio_loss = du.all_reduce([audio_loss])[0]
                valid_visual = du.all_gather([torch.IntTensor(valid_visual)]).sum()
                valid_audio = du.all_gather([torch.IntTensor(valid_audio)]).sum()

            val_meter.update(
                    verb_preds.detach().cpu(),
                    noun_preds.detach().cpu(),
                    action_preds.detach().cpu(),
                    aud_preds.detach().cpu(),
                    v_action_ids.detach().cpu(),
                    a_action_ids.detach().cpu(),
                    v_target.detach().cpu(),
                    a_target.detach().cpu(),
                    visual_loss.item(),
                    visual_loss_verb.item(),
                    visual_loss_noun.item(),
                    visual_loss_action.item(),
                    audio_loss.item(),
                    valid_visual,
                    valid_audio
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

        best_acc1, is_best, stop = val_meter.update_epoch(epoch)

        # Log validation epoch stats
        if wandb_log and is_master_proc:
            iters += 1
            log_dict = val_meter.get_val_epoch_stats(iters)
            wandb.log(log_dict)
        logger.info(val_meter.get_val_epoch_message(epoch))

        val_meter.reset()

        return best_acc1, is_best, stop, iters