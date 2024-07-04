import numpy as np
import datetime
import torch

from fvcore.common.timer import Timer

import time_interval_machine.utils.logging as logging
import time_interval_machine.utils.misc as misc

logger = logging.get_logger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TrainMeter(object):
    """Tracks multiple metrics for TIM model during training"""
    def __init__(self, args):
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()

        self.losses = AverageMeter()
        self.drloc_losses = AverageMeter()
        self.visual_action_losses = AverageMeter()
        self.visual_reg_losses = AverageMeter()
        self.visual_losses = AverageMeter()
        self.positive_losses = AverageMeter()
        self.negative_losses = AverageMeter()

        self.audio_losses = AverageMeter()
        self.audio_reg_losses = AverageMeter()

        self.modality = args.data_modality
        self.include_dr_loc = args.lambda_drloc > 0.0
        self.include_verb_noun = args.include_verb_noun

        if self.include_verb_noun:
            self.visual_verb_losses = AverageMeter()
            self.visual_noun_losses = AverageMeter()


        self.reset()

    def reset(self):
        self.losses.reset()
        self.drloc_losses.reset()
        self.visual_action_losses.reset()
        self.visual_reg_losses.reset()
        self.visual_losses.reset()
        self.audio_losses.reset()

        if self.include_verb_noun:
            self.visual_verb_losses.reset()
            self.visual_noun_losses.reset()

        self.positive_losses.reset()
        self.negative_losses.reset()


    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def net_toc(self):
        self.net_timer.pause()

    def update(
            self,
            visual_loss,
            visual_loss_verb,
            visual_loss_noun,
            visual_loss_action,
            visual_reg_loss,
            audio_loss,
            audio_reg_loss,
            loss,
            drloc_loss,
            positive_loss,
            negative_loss,
            visual_cls_num,
            audio_cls_num,
            visual_reg_num,
            audio_reg_num
        ):

        # Track visual loss
        if visual_cls_num > 0 and "visual" in self.modality:
            self.visual_losses.update(visual_loss, visual_cls_num)

            if self.include_verb_noun:
                self.visual_verb_losses.update(visual_loss_verb, visual_cls_num)
                self.visual_noun_losses.update(visual_loss_noun, visual_cls_num)

            self.visual_action_losses.update(visual_loss_action, visual_cls_num)
            self.visual_reg_losses.update(visual_reg_loss, visual_reg_num)
            self.positive_losses.update(positive_loss, visual_reg_num)
            self.negative_losses.update(negative_loss, (visual_cls_num - visual_reg_num))


        # Track audio loss
        if audio_cls_num > 0 and "audio" in self.modality:
            self.audio_losses.update(audio_loss, audio_cls_num)
            self.audio_reg_losses.update(audio_reg_loss, audio_reg_num)
            self.positive_losses.update(positive_loss, audio_reg_num)
            self.negative_losses.update(negative_loss, (audio_cls_num - audio_reg_num))


        # Track overall loss and localization loss
        self.losses.update(loss, (visual_cls_num + audio_cls_num))
        self.drloc_losses.update(drloc_loss, (visual_cls_num + audio_cls_num))

    def get_train_stats(self, lr, training_iterations, normaliser, num_pos, num_neg):
        stats_dict = {
                    "Train/lr": lr,
                    "Train/normaliser": normaliser,
                    "Train/positives": num_pos,
                    "Train/negatives": num_neg,
                    "Train/positive_ratio": num_pos / (num_pos + num_neg),
                    "train_step": training_iterations
                }

        if self.include_dr_loc:
            stats_dict.update({"Train/drloc_loss": self.drloc_losses.avg})

        if "visual" in self.modality:
            if self.include_verb_noun:
                stats_dict.update(
                    {
                        "Train/verb/visual_loss": self.visual_verb_losses.avg,
                        "Train/noun/visual_loss": self.visual_noun_losses.avg
                    }
                )
            stats_dict.update(
                    {
                        "Train/visual_loss": self.visual_losses.avg,
                        "Train/visual_action_loss": self.visual_action_losses.avg,
                        "Train/visual_positive_loss": self.positive_losses.avg,
                        "Train/visual_negative_loss": self.negative_losses.avg,
                        "Train/visual_reg_loss": self.visual_reg_losses.avg,
                    }
                )
        if "audio" in self.modality:
            stats_dict.update(
                    {
                        "Train/audio_loss": self.audio_losses.avg,
                        "Train/audio_positive_loss": self.positive_losses.avg,
                        "Train/audio_negative_loss": self.negative_losses.avg,
                        "Train/audio_reg_loss": self.audio_reg_losses.avg,
                    }
                )

        return stats_dict

    def get_train_message(self, epoch, i, dataloader_size, lr):
        message_str = ('| Epoch: [{0}][{1}/{2}] | lr: {lr:.5f} |'
                    ' Time: {batch_time:.3f} |'
                    ' Data: {data_time:.3f} |'
                    ' Net: {net_time:.3f} |'.format(
                                        epoch+1,
                                        i,
                                        dataloader_size,
                                        lr=lr,
                                        batch_time=self.iter_timer.seconds(),
                                        data_time=self.data_timer.seconds(),
                                        net_time=self.net_timer.seconds()
                                    )
                    )
        if "visual" in self.modality:
            message_str += (' Visual Views Seen: {visual_loss.count} ({positive_loss.count}, {negative_loss.count}) |'
                            ' Visual Loss: {visual_loss.avg:.4f} |'
                            ' Label Loss: ({positive_loss.avg:.4f}, {negative_loss.avg:.4f}) |'
                            ' Visual Reg Loss: {visual_reg_loss.avg:.4f} |'.format(
                                    visual_loss=self.visual_losses,
                                    positive_loss=self.positive_losses,
                                    negative_loss=self.negative_losses,
                                    visual_reg_loss=self.visual_reg_losses,
                                )
                        )
        if "audio" in self.modality:
            message_str += (' Audio Views Seen: {audio_loss.count} ({positive_loss.count}, {negative_loss.count}) |'
                            ' Audio Loss: {audio_loss.avg:.4f} |'
                            ' Label Loss: ({positive_loss.avg:.4f}, {negative_loss.avg:.4f}) |'
                            ' Audio Reg Loss: {audio_reg_loss.avg:.4f} |'.format(
                                    audio_loss=self.audio_losses,
                                    positive_loss=self.positive_losses,
                                    negative_loss=self.negative_losses,
                                    audio_reg_loss=self.audio_reg_losses,
                                )
                        )
        if self.include_dr_loc:
            message_str += ' DRL Loss: {drloc_loss.avg:.4f} |'.format(
                                    drloc_loss=self.drloc_losses
                                )

        message_str += (' Loss: {loss.avg:.4f} |'
                        ' RAM: {ram[0]:.2f}/{ram[1]:.2f}GB |'
                        ' GPU: {gpu[0]:.2f}/{gpu[1]:.2f}GB |'.format(
                                                loss=self.losses,
                                                ram=misc.cpu_mem_usage(),
                                                gpu=misc.gpu_mem_usage()
                                            )
                        )
        return message_str

    def get_train_epoch_stats(self, iters):
        stats_dict = {
                    "train_step": iters
                }

        if "visual" in self.modality:
            if self.include_verb_noun:
                stats_dict.update(
                    {
                        "Train_Epoch/visual/verb_loss": self.visual_verb_losses.avg,
                        "Train_Epoch/visual/noun_loss": self.visual_noun_losses.avg
                    }
                )
            stats_dict.update(
                    {
                        "Train_Epoch/visual/loss": self.visual_losses.avg,
                        "Train_Epoch/visual/action_loss": self.visual_action_losses.avg,
                        "Train_Epoch/visual/positive_loss": self.positive_losses.avg,
                        "Train_Epoch/visual/negative_loss": self.negative_losses.avg,
                        "Train_Epoch/visual/reg_loss": self.visual_reg_losses.avg,
                    }
                )
        if "audio" in self.modality:
            stats_dict.update(
                    {
                        "Train_Epoch/audio/loss": self.audio_losses.avg,
                        "Train_Epoch/audio/positive_loss": self.positive_losses.avg,
                        "Train_Epoch/audio/negative_loss": self.negative_losses.avg,
                        "Train_Epoch/audio/reg_loss": self.audio_reg_losses.avg
                    }
                )

        return stats_dict

    def get_train_epoch_message(self, epoch):
        message_str = (f'\nEpoch {epoch+1} Results:\n' \
                '\t====================================================\n')

        if "visual" in self.modality:
            if self.include_verb_noun:
                message_str += (f'\tVisual Views Seen: {self.visual_losses.count} ' \
                    f'({self.positive_losses.count}, {self.negative_losses.count})\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tVisual Verb Loss {self.visual_verb_losses.avg:.3f}\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tVisual Noun Loss {self.visual_noun_losses.avg:.3f}\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tVisual Action Loss {self.visual_action_losses.avg:.3f}\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tVisual Loss {self.visual_losses.avg:.5f}\n' \
                    f'\tVisual Reg Loss {self.visual_reg_losses.avg:.5f}\n' \
                    '\t====================================================\n')
            else:
                message_str += (f'\tVisual Views Seen: {self.visual_losses.count} ' \
                    f'({self.positive_losses.count}, {self.negative_losses.count})\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tVisual Action Loss {self.visual_action_losses.avg:.3f}\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tLabel Loss ({self.positive_losses.avg:.4f}, {self.negative_losses.avg:.4f})\n' \
                    f'\tVisual Loss {self.visual_losses.avg:.5f}\n' \
                    f'\tVisual Reg Loss {self.visual_reg_losses.avg:.5f}\n' \
                    '\t====================================================\n')
        if "audio" in self.modality:
            message_str += (f'\tAudio Views Seen: {self.audio_losses.count} ' \
                f'({self.positive_losses.count}, {self.negative_losses.count})\n' \
                '\t----------------------------------------------------\n' \
                f'\tLabel Loss ({self.positive_losses.avg:.4f}, {self.negative_losses.avg:.4f})\n' \
                f'\tAudio Loss {self.audio_losses.avg:.5f}\n' \
                f'\tAudio Reg Loss {self.audio_reg_losses.avg:.5f}\n' \
                '\t====================================================\n')

        if self.include_dr_loc:
            message_str += f'\tDR Loc Loss {self.drloc_losses.avg:.5f}\n'

        message_str += (f'\tLoss {self.losses.avg:.5f}\n' \
                        '\t====================================================')

        return message_str

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class InferenceMeter(object):
    """Tracks multiple metrics for TIM model during validation"""
    def __init__(self, args):
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()

        self.losses = AverageMeter()
        self.visual_action_losses = AverageMeter()
        self.visual_reg_losses = AverageMeter()
        self.visual_losses = AverageMeter()
        self.positive_losses = AverageMeter()
        self.negative_losses = AverageMeter()

        self.audio_losses = AverageMeter()
        self.audio_reg_losses = AverageMeter()

        self.modality = args.data_modality
        self.include_verb_noun = args.include_verb_noun

        if self.include_verb_noun:
            self.visual_verb_losses = AverageMeter()
            self.visual_noun_losses = AverageMeter()

        self.best_vis_loss = float("inf")
        self.best_aud_loss = float("inf")
        self.num_positives = 0
        self.num_negatives = 0


        self.reset()

    def reset(self):
        self.losses.reset()
        self.visual_action_losses.reset()
        self.visual_reg_losses.reset()
        self.visual_losses.reset()
        self.audio_losses.reset()

        if self.include_verb_noun:
            self.visual_verb_losses.reset()
            self.visual_noun_losses.reset()
            
        self.positive_losses.reset()
        self.negative_losses.reset()

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def net_toc(self):
        self.net_timer.pause()

    def update(
            self,
            visual_loss,
            visual_loss_verb,
            visual_loss_noun,
            visual_loss_action,
            visual_reg_loss,
            audio_loss,
            audio_reg_loss,
            loss,
            positive_loss,
            negative_loss,
            visual_cls_num,
            audio_cls_num,
            visual_reg_num,
            audio_reg_num
        ):

        # Track visual loss
        if visual_cls_num > 0 and "visual" in self.modality:
            self.visual_losses.update(visual_loss, visual_cls_num)

            if self.include_verb_noun:
                self.visual_verb_losses.update(visual_loss_verb, visual_cls_num)
                self.visual_noun_losses.update(visual_loss_noun, visual_cls_num)

            self.visual_action_losses.update(visual_loss_action, visual_cls_num)
            self.visual_reg_losses.update(visual_reg_loss, visual_reg_num)
            self.positive_losses.update(positive_loss, visual_reg_num)
            self.negative_losses.update(negative_loss, (visual_cls_num - visual_reg_num))


        # Track audio loss
        if audio_cls_num > 0 and "audio" in self.modality:
            self.audio_losses.update(audio_loss, audio_cls_num)
            self.audio_reg_losses.update(audio_reg_loss, audio_reg_num)
            self.positive_losses.update(positive_loss, audio_reg_num)
            self.negative_losses.update(negative_loss, (audio_cls_num - audio_reg_num))

        # Track overall loss and localization loss
        self.losses.update(loss, (visual_cls_num + audio_cls_num))

    def update_epoch(self, epoch):
        is_best_visual = self.visual_losses.avg < self.best_vis_loss
        is_best_audio = self.audio_losses.avg < self.best_aud_loss

        best_loss = {
                        "visual": self.best_vis_loss,
                        "audio": self.best_aud_loss
                    }

        is_best = "none"
        if is_best_visual:
            is_best = "visual" if is_best_visual else ""
            self.best_vis_loss = min(self.visual_losses.avg, self.best_vis_loss)
            self.last_best_epoch = epoch
        if is_best_audio:
            is_best = "audio_" + is_best if "visual" in is_best else "audio"
            self.best_aud_loss = min(self.audio_losses.avg, self.best_aud_loss)
            self.last_best_epoch = epoch

        return best_loss, is_best

    def get_val_message(self, epoch, i, dataloader_size):
        message_str = ('| Epoch: [{0}][{1}/{2}] |'
                    ' Time: {batch_time:.3f} |'
                    ' Data: {data_time:.3f} |'
                    ' Net: {net_time:.3f} |'.format(
                                        epoch+1,
                                        i,
                                        dataloader_size,
                                        batch_time=self.iter_timer.seconds(),
                                        data_time=self.data_timer.seconds(),
                                        net_time=self.net_timer.seconds()
                                    )
                    )
        if "visual" in self.modality:
            message_str += (' Visual Views Seen: {visual_loss.count} ({positive_loss.count}, {negative_loss.count}) |'
                            ' Visual Loss: {visual_loss.avg:.4f} |'
                            ' Label Loss: ({positive_loss.avg:.4f}, {negative_loss.avg:.4f}) |'
                            ' Visual Reg Loss: {visual_reg_loss.avg:.4f} |'.format(
                                    visual_loss=self.visual_losses,
                                    positive_loss=self.positive_losses,
                                    negative_loss=self.negative_losses,
                                    visual_reg_loss=self.visual_reg_losses,
                                )
                        )
        if "audio" in self.modality:
            message_str += (' Audio Views Seen: {audio_loss.count} ({positive_loss.count}, {negative_loss.count}) |'
                            ' Audio Loss: {audio_loss.avg:.4f} |'
                            ' Label Loss: ({positive_loss.avg:.4f}, {negative_loss.avg:.4f}) |'
                            ' Audio Reg Loss: {audio_reg_loss.avg:.4f} |'.format(
                                    audio_loss=self.audio_losses,
                                    positive_loss=self.positive_losses,
                                    negative_loss=self.negative_losses,
                                    audio_reg_loss=self.audio_reg_losses,
                                )
                        )

        message_str += (' Loss: {loss.avg:.4f} |'
                        ' RAM: {ram[0]:.2f}/{ram[1]:.2f}GB |'
                        ' GPU: {gpu[0]:.2f}/{gpu[1]:.2f}GB |'.format(
                                                loss=self.losses,
                                                ram=misc.cpu_mem_usage(),
                                                gpu=misc.gpu_mem_usage()
                                            )
                        )
        return message_str

    def get_val_epoch_stats(self, iters):
        stats_dict = {
                    "val_step": iters
                }

        if "visual" in self.modality:
            if self.include_verb_noun:
                stats_dict.update(
                    {
                        "Val/visual/verb_loss": self.visual_verb_losses.avg,
                        "Val/visual/noun_loss": self.visual_noun_losses.avg
                    }
                )
            stats_dict.update(
                    {
                        "Val/visual/loss": self.visual_losses.avg,
                        "Val/visual/action_loss": self.visual_action_losses.avg,
                        "Val/visual/positive_loss": self.positive_losses.avg,
                        "Val/visual/negative_loss": self.negative_losses.avg,
                        "Val/visual/reg_loss": self.visual_reg_losses.avg,
                    }
                )
        if "audio" in self.modality:
            stats_dict.update(
                    {
                        "Val/audio/loss": self.audio_losses.avg,
                        "Val/audio/positive_loss": self.positive_losses.avg,
                        "Val/audio/negative_loss": self.negative_losses.avg,
                        "Val/audio/reg_loss": self.audio_reg_losses.avg
                    }
                )

        return stats_dict

    def get_val_epoch_message(self, epoch):
        message_str = (f'\nEpoch {epoch+1} Results:\n' \
                '\t====================================================\n')

        if "visual" in self.modality:
            if self.include_verb_noun:
                message_str += (f'\tVisual Views Seen: {self.visual_losses.count} ' \
                    f'({self.positive_losses.count}, {self.negative_losses.count})\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tVisual Verb Loss {self.visual_verb_losses.avg:.3f}\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tVisual Noun Loss {self.visual_noun_losses.avg:.3f}\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tVisual Action Loss {self.visual_action_losses.avg:.3f}\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tVisual Loss {self.visual_losses.avg:.5f}\n' \
                    f'\tVisual Reg Loss {self.visual_reg_losses.avg:.5f}\n' \
                    '\t====================================================\n')
            else:
                message_str += (f'\tVisual Views Seen: {self.visual_losses.count} ' \
                    f'({self.positive_losses.count}, {self.negative_losses.count})\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tVisual Action Loss {self.visual_action_losses.avg:.3f}\n' \
                    '\t----------------------------------------------------\n' \
                    f'\tLabel Loss ({self.positive_losses.avg:.4f}, {self.negative_losses.avg:.4f})\n' \
                    f'\tVisual Loss {self.visual_losses.avg:.5f}\n' \
                    f'\tVisual Reg Loss {self.visual_reg_losses.avg:.5f}\n' \
                    '\t====================================================\n')
        if "audio" in self.modality:
            message_str += (f'\tAudio Views Seen: {self.audio_losses.count} ' \
                f'({self.positive_losses.count}, {self.negative_losses.count})\n' \
                '\t----------------------------------------------------\n' \
                f'\tAudio Loss {self.audio_losses.avg:.5f}\n' \
                f'\tLabel Loss ({self.positive_losses.avg:.4f}, {self.negative_losses.avg:.4f})\n' \
                f'\tAudio Reg Loss {self.audio_reg_losses.avg:.5f}\n' \
                '\t====================================================\n')

        message_str += (f'\tLoss {self.losses.avg:.5f}\n' \
                        '\t====================================================')

        return message_str

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class FeatureMeter(object):
    """Tracks multiple metrics for TIM model during validation"""
    def __init__(
            self,
            args
        ):
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.total_time = 0
        self.peak_cpu_mem = 0.0
        self.peak_gpu_mem = 0.0
        self.modality = args.data_modality
        self.include_verb_noun = args.include_verb_noun
        self.num_classes = args.num_class

        # Initialize tensors.
        if self.include_verb_noun:
            self.verb_preds = torch.empty((0, self.num_classes[0][0]))
            self.noun_preds = torch.empty((0, self.num_classes[0][1]))
            self.action_preds = torch.empty((0, self.num_classes[0][2]))
        else:
            self.action_preds = torch.empty((0, self.num_classes[0]))

        self.aud_preds = torch.empty((0, self.num_classes[1]))

        self.v_props = torch.empty((0, 2))
        self.a_props = torch.empty((0, 2))

        self.og_v_props = torch.empty((0, 2))
        self.og_a_props = torch.empty((0, 2))

        self.video_ids = np.zeros(0, dtype=object)

    def reset(self):
        """
        Reset the metric.
        """
        if self.include_verb_noun:
            self.verb_preds = torch.empty((0, self.num_classes[0][0]))
            self.noun_preds = torch.empty((0, self.num_classes[0][1]))
            self.action_preds = torch.empty((0, self.num_classes[0][2]))
        else:
            self.action_preds = torch.empty((0, self.num_classes[0]))

        self.aud_preds = torch.empty((0, self.num_classes[1]))

        self.v_props = torch.empty((0, 2))
        self.a_props = torch.empty((0, 2))

        self.og_v_props = torch.empty((0, 2))
        self.og_a_props = torch.empty((0, 2))

        self.video_ids = np.zeros(0, dtype=object)


    def iter_tic(self):
        """
        Start to record time.
        """
        self.total_time += self.iter_timer.seconds()
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def net_toc(self):
        self.net_timer.pause()


    def update(self, features, regressions, query_times, metadata):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            features (tensor): features from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (feature_dim).
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        ram = misc.cpu_mem_usage()
        gpu = misc.gpu_mem_usage()

        self.peak_cpu_mem = max(ram[0], self.peak_cpu_mem)
        self.peak_gpu_mem = max(gpu[0], self.peak_gpu_mem)

        if "visual" in self.modality:
            video_ids = np.array(metadata['video_id'], dtype=object)
            num_queries = int(features[2].shape[0] / video_ids.shape[0])
            video_ids = np.repeat(video_ids, num_queries)

            win_starts = metadata['window_start'].repeat_interleave(num_queries)
            win_size = metadata['window_size'][0]


            v_query_times = query_times[0].cpu()
            v_query_times = torch.flatten(v_query_times, end_dim=-2)
            max_time = v_query_times.max()

            og_query = (v_query_times * win_size) + win_starts[:, None]
            self.og_v_props = torch.concat([self.og_v_props, og_query])

            v_proposals = regressions[0].cpu()
            v_proposals = torch.clamp(v_proposals, min=0.0, max=max_time)
            v_proposals = (v_proposals * win_size) + win_starts[:, None]
            self.v_props = torch.concat([self.v_props, v_proposals])

            if self.include_verb_noun:
                verb_preds = torch.sigmoid(features[0]).cpu()
                noun_preds = torch.sigmoid(features[1]).cpu()
                self.verb_preds = torch.concat([self.verb_preds, verb_preds])
                self.noun_preds = torch.concat([self.noun_preds, noun_preds])

            action_preds = torch.sigmoid(features[2]).cpu()
            self.action_preds = torch.concat([self.action_preds, action_preds])
            self.video_ids = np.concatenate([self.video_ids, video_ids], axis=0)

        if "audio" in self.modality:
            video_ids = np.array(metadata['video_id'], dtype=object)
            num_queries = int(features[3].shape[0] / video_ids.shape[0])
            video_ids = np.repeat(video_ids, num_queries)

            win_starts = metadata['window_start'].repeat_interleave(num_queries)
            win_size = metadata['window_size'][0]


            a_query_times = query_times[1].cpu()
            a_query_times = torch.flatten(a_query_times, end_dim=-2)
            max_time = a_query_times.max()

            og_query = (a_query_times * win_size) + win_starts[:, None]
            self.og_a_props = torch.concat([self.og_a_props, og_query])

            a_proposals = regressions[1].cpu()
            a_proposals = torch.clamp(a_proposals, min=0.0, max=max_time)
            a_proposals = (a_proposals * win_size) + win_starts[:, None]
            self.a_props = torch.concat([self.a_props, a_proposals])

            aud_preds = torch.sigmoid(features[3]).cpu()

            self.aud_preds = torch.concat([self.aud_preds, aud_preds])
            self.video_ids = np.concatenate([self.video_ids, video_ids], axis=0)


    def get_feat_message(self, iter, total_iters):
        return ('| Iter: [{0}]/[{1}] |'
                ' Features Extracted: {num_feats} |'
                ' Time: {batch_time:.3f} |'
                ' Data: {data_time:.3f} |'
                ' Net: {net_time:.3f} |'
                ' RAM: {ram[0]:.2f}/{ram[1]:.2f} GB |'
                ' GPU: {gpu[0]:.2f}/{gpu[1]:.2f} GB |'.format(
                                            iter,
                                            total_iters,
                                            num_feats=(self.action_preds.size(0) + self.aud_preds.size(0)),
                                            batch_time=self.iter_timer.seconds(),
                                            data_time=self.data_timer.seconds(),
                                            net_time=self.net_timer.seconds(),
                                            ram=misc.cpu_mem_usage(),
                                            gpu=misc.gpu_mem_usage()
                                        )
            )

    def save_chunk(self):
        data = {"video_ids": self.video_ids}
        if "visual" in self.modality:
            if self.include_verb_noun:
                data.update(
                    {
                        "verb": self.verb_preds.numpy(),
                        "noun": self.noun_preds.numpy()
                    }
                )
            data.update(
                    {
                        "action": self.action_preds.numpy(),
                        "v_proposals": self.v_props.numpy(),
                        "og_v_props": self.og_v_props.numpy()
                    }
                )

        if "audio" in self.modality:
            data.update(
                    {
                        "audio": self.aud_preds.numpy(),
                        "a_proposals": self.a_props.numpy(),
                        "og_a_props": self.og_a_props.numpy()
                    }
                )
        self.reset()
        return data


    def finalize_metrics(self):
        data = {"video_ids": self.video_ids}
        if "visual" in self.modality:
            if self.include_verb_noun:
                data.update(
                    {
                        "verb": self.verb_preds.numpy(),
                        "noun": self.noun_preds.numpy()
                    }
                )
            data.update(
                    {
                        "action": self.action_preds.numpy(),
                        "v_proposals": self.v_props.numpy(),
                        "og_v_props": self.og_v_props.numpy()
                    }
                )

        if "audio" in self.modality:
            data.update(
                    {
                        "audio": self.aud_preds.numpy(),
                        "a_proposals": self.a_props.numpy(),
                        "og_a_props": self.og_a_props.numpy()
                    }
                )
        time_taken = str(datetime.timedelta(seconds=int(self.total_time)))
        final_message = (f'| Features Extracted: {(self.action_preds.size(0) + self.aud_preds.size(0))} |'
                        f' Time Elapsed: {time_taken} |'
                        f' Peak RAM: {self.peak_cpu_mem:.2f} GB |'
                        f' Peak GPU: {self.peak_gpu_mem:.2f} GB |'
                    )
        logger.info(final_message)
        return data
