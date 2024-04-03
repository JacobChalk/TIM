import numpy as np
import random
import pickle
import torch
import os

import time_interval_machine.datasets.loader as loader
import time_interval_machine.utils.logging as logging
import time_interval_machine.utils.distributed as du
import time_interval_machine.utils.checkpoint as ch
import time_interval_machine.utils.misc as misc

from time_interval_machine.utils.meters import FeatureMeter
from time_interval_machine.models.build import build_model

torch.set_printoptions(sci_mode=False)
logger = logging.get_logger(__name__)

torch.backends.cudnn.benchmark = True

def init_extract(args):
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

    model, args = build_model(args)
    logger.info(model)

    logger.info("Output dir : {}".format(args.output_dir))

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

    feat_loader = loader.create_loader(args, mode, args.data_modality, rng_generator)
    feat_meter = FeatureMeter(
                num_actions=feat_loader.dataset.num_actions,
                args=args
            )

    logger.info(f"Extracting features for {feat_loader.dataset.num_actions} actions across {len(feat_loader.dataset)} windows.")
    extract_features(
                args=args,
                feat_loader=feat_loader,
                model=model,
                is_master_proc=is_master_proc,
                feat_meter=feat_meter
            )

def extract_features(
            args,
            feat_loader,
            model,
            is_master_proc,
            feat_meter
        ):
    with torch.no_grad():
        # Switch to evaluate mode
        model.eval()
        is_master_proc = du.is_master_proc(args.num_gpus * args.num_shards)
        feat_meter.iter_tic()
        for i, (visual_input, audio_input, times, target, metadata) in enumerate(feat_loader):
            # Put data onto GPU
            visual_input = visual_input.cuda(non_blocking=True)
            audio_input = audio_input.cuda(non_blocking=True)
            times = times.cuda(non_blocking=True)
            target = {k: v.cuda(non_blocking=True) for k, v in target.items()}

            metadata, v_queries, a_queries = misc.process_metadata(metadata)

            target = {k: torch.flatten(v) for k, v in target.items()}

            # Measure data loading time
            feat_meter.data_toc()

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
            feat_meter.net_toc()

            # Gather metrics onto 1 GPU
            if args.num_gpus > 1:
                output = du.all_gather([output])

                meta = du.all_gather_unaligned(meta)
                key = list(meta[0].keys())[0]
                metadata = {key: []}
                for i in range(len(meta)):
                    metadata[key].extend(meta[i][key])

            feat_meter.update(
                    output[0],
                    metadata
                )

            # Measure elapsed time
            feat_meter.iter_toc()
            if ((i+1) % args.print_freq == 0 or i == 0) and is_master_proc:
                message = feat_meter.get_feat_message(i, len(feat_loader))
                logger.info(message)

            feat_meter.iter_tic()

        data = feat_meter.finalize_metrics()

        if is_master_proc:
            features_dir = os.path.join(args.output_dir, 'features')
            if not os.path.exists(features_dir):
                os.makedirs(features_dir)
            feats_file = str(args.video_val_action_pickle).split("/")[-1].replace(".pkl", "")
            file_path = os.path.join(features_dir, f'{feats_file}_features.pkl')
            logger.info(f"Saving to file path: {file_path}")
            pickle.dump(data, open(file_path, 'wb'))

