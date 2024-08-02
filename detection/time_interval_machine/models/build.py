import torch

from time_interval_machine.models.tim import TIM

def build_model(args, gpu_id=None):
    """
    Builds the video model.
    Args:
        args: args that contains the hyper-parameters to build the backbone.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            args.num_gpus <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            args.num_gpus == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    model = TIM(
                args.num_class,
                visual_input_dim=args.visual_input_dim,
                audio_input_dim=args.audio_input_dim,
                feat_drop=args.feat_dropout,
                seq_drop=args.seq_dropout,
                d_model=args.d_model,
                feedfoward_scale=args.feedfoward_scale,
                nhead=args.nhead,
                num_layers=args.num_layers,
                enc_dropout=args.enc_dropout,
                input_modality=args.model_modality,
                data_modality=args.data_modality,
                num_feats=args.num_feats,
                include_verb_noun=args.include_verb_noun,
                iou_threshold=args.iou_threshold,
                label_smoothing=args.label_smoothing
            )

    if args.num_gpus:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
                # Transfer the model to the current GPU device
        if args.num_gpus > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(device=cur_device)

    # Use multi-process data parallel model in the multi-gpu setting
    if args.num_gpus > 1:
        # Divide workers evengly amongst gpus
        if args.workers < args.num_gpus or args.workers == 0:
            args.workers = 0
        else:
            args.workers = int((args.workers + args.num_gpus - 1) / args.num_gpus)
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            find_unused_parameters=False
        )

    return model, args
