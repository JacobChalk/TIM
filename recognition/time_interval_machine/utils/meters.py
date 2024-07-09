import numpy as np
import datetime
import torch

from fvcore.common.timer import Timer

from time_interval_machine.utils.metrics import accuracy, multitask_accuracy
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
    def __init__(self, args, num_actions):
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()

        self.losses = AverageMeter()
        self.drloc_losses = AverageMeter()
        self.visual_verb_losses = AverageMeter()
        self.visual_noun_losses = AverageMeter()
        self.visual_action_losses = AverageMeter()
        self.visual_losses = AverageMeter()

        self.audio_losses = AverageMeter()

        self.dataset = args.dataset
        self.modality = args.data_modality
        self.include_dr_loc = args.lambda_drloc > 0.0
        self.include_verb_noun = args.include_verb_noun

        # Initialize tensors.
        pred_precision = torch.float16 if args.enable_amp else torch.float32
        if self.include_verb_noun:
            self.verb_preds = torch.zeros((num_actions, args.num_class[0][0]), dtype=pred_precision)
            self.noun_preds = torch.zeros((num_actions, args.num_class[0][1]), dtype=pred_precision)
            self.action_preds = torch.zeros((num_actions, args.num_class[0][2]), dtype=pred_precision)
        else:
            self.action_preds = torch.zeros((num_actions, args.num_class[0]), dtype=pred_precision)

        self.aud_preds = torch.zeros((num_actions, args.num_class[1]), dtype=pred_precision)
        self.seen_count = torch.zeros((num_actions,), dtype=torch.float32)
        self.v_labels = torch.full(size=(num_actions, 3), fill_value=-1, dtype=torch.int32)
        self.a_labels = torch.full(size=(num_actions,), fill_value=-1, dtype=torch.int32)

        self.verb_acc = (0.0, 0.0)
        self.noun_acc = (0.0, 0.0)
        self.action_acc = (0.0, 0.0)
        self.mt_action_acc = (0.0, 0.0)
        self.aud_acc = (0.0, 0.0)
        self.combined_acc = (0.0, 0.0)

        self.reset()

    def reset(self):
        self.losses.reset()
        self.drloc_losses.reset()
        self.visual_verb_losses.reset()
        self.visual_noun_losses.reset()
        self.visual_action_losses.reset()
        self.visual_losses.reset()
        self.audio_losses.reset()

        if self.include_verb_noun:
            self.verb_preds.zero_()
            self.noun_preds.zero_()
        self.action_preds.zero_()
        self.aud_preds.zero_()
        self.seen_count.zero_()

        self.verb_acc = (0.0, 0.0)
        self.noun_acc = (0.0, 0.0)
        self.mt_action_acc = (0.0, 0.0)

        self.action_acc = (0.0, 0.0)
        self.aud_acc = (0.0, 0.0)
        self.combined_acc = (0.0, 0.0)


    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def net_toc(self):
        self.net_timer.pause()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update(
            self,
            verb_preds,
            noun_preds,
            action_preds,
            aud_preds,
            v_action_ids,
            a_action_ids,
            v_labels,
            a_labels,
            visual_loss,
            visual_loss_verb,
            visual_loss_noun,
            visual_loss_action,
            audio_loss,
            loss,
            drloc_loss,
            valid_visual,
            valid_audio
        ):

        # Track visual loss
        if valid_visual > 0 and "visual" in self.modality:
            self.visual_losses.update(visual_loss, valid_visual)

            if self.include_verb_noun:
                self.visual_verb_losses.update(visual_loss_verb, valid_visual)
                self.visual_noun_losses.update(visual_loss_noun, valid_visual)
                self.verb_preds.index_add_(dim=0, index=v_action_ids, source=verb_preds)
                self.noun_preds.index_add_(dim=0, index=v_action_ids, source=noun_preds)

            self.visual_action_losses.update(visual_loss_action, valid_visual)
            self.action_preds.index_add_(dim=0, index=v_action_ids, source=action_preds)
            self.seen_count.index_add_(dim=0, index=v_action_ids, source=torch.ones_like(v_action_ids).float())

            self.v_labels[v_action_ids] = v_labels.int()

        # Track audio loss
        if valid_audio > 0 and "audio" in self.modality:
            self.audio_losses.update(audio_loss, valid_audio)

            self.aud_preds.index_add_(dim=0, index=a_action_ids, source=aud_preds)
            self.seen_count.index_add_(dim=0, index=a_action_ids, source=torch.ones_like(a_action_ids).float())

            self.a_labels[a_action_ids] = a_labels.int()

        # Track overall loss and localization loss
        self.losses.update(loss, (valid_visual + valid_audio))
        self.drloc_losses.update(drloc_loss, (valid_visual + valid_audio))

    def get_train_stats(self, lr, training_iterations):
        stats_dict = {
                    "Train/lr": lr,
                    "train_step": training_iterations
                }

        if self.include_dr_loc:
            stats_dict.update({"Train/drloc_loss": self.drloc_losses.avg})

        if "visual" in self.modality:
            if self.include_verb_noun:
                stats_dict.update(
                    {
                        "Train/visual/verb_Top1_acc": self.verb_acc[0],
                        "Train/visual/verb_Top5_acc": self.verb_acc[1],
                        "Train/visual/noun_Top1_acc": self.noun_acc[0],
                        "Train/visual/noun_Top5_acc": self.noun_acc[1],
                        "Train/verb/visual_loss": self.visual_verb_losses.avg,
                        "Train/noun/visual_loss": self.visual_noun_losses.avg,
                        "Train/visual/Top1_acc": self.mt_action_acc[0],
                        "Train/visual/Top5_acc": self.mt_action_acc[1],
                    }
                )
            stats_dict.update(
                    {
                        "Train/visual_loss": self.visual_losses.avg,
                        "Train/visual_action_loss": self.visual_action_losses.avg,
                        "Train/visual/act_Top1_acc": self.action_acc[0],
                        "Train/visual/act_Top5_acc": self.action_acc[1]
                    }
                )
        if "audio" in self.modality:
            stats_dict.update(
                    {
                        "Train/audio_loss": self.audio_losses.avg,
                        "Train/audio/Top1_acc": self.aud_acc[0],
                        "Train/audio/Top5_acc": self.aud_acc[1],
                    }
                )

        return stats_dict

    def get_train_message(self, epoch, i, dataloader_size, lr):
        message_str = ('| Epoch: [{0}][{1}/{2}] |'
                    ' lr: {lr:.5f} |'
                    ' Time: {batch_time:.3f} |'
                    ' Data: {data_time:.3f} |'
                    ' Net: {net_time:.3f} |'.format(
                                        epoch+1,
                                        i+1,
                                        dataloader_size,
                                        lr=lr,
                                        batch_time=self.iter_timer.seconds(),
                                        data_time=self.data_timer.seconds(),
                                        net_time=self.net_timer.seconds()
                                    )
                    )
        if "visual" in self.modality:
            message_str += (' Visual Views Seen: {visual_loss.count} |'
                            ' Visual Loss: {visual_loss.avg:.4f} |'.format(
                                    visual_loss=self.visual_losses
                                )
                        )
        if "audio" in self.modality:
            message_str += (' Audio Views Seen: {audio_loss.count} |'
                            ' Audio Loss: {audio_loss.avg:.4f} |'.format(
                                    audio_loss=self.audio_losses
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

    def update_epoch(self):
        if "visual" in self.modality:
            valid_indices = (self.v_labels[:, 2] != -1)
            seen = self.seen_count[valid_indices].unsqueeze(1)

            if self.include_verb_noun:
                verb_preds = (self.verb_preds[valid_indices].float() / seen.repeat(1, self.verb_preds.size(1))).softmax(dim=1)
                noun_preds = (self.noun_preds[valid_indices].float() / seen.repeat(1, self.noun_preds.size(1))).softmax(dim=1)
                verb_labels = self.v_labels[valid_indices, 0]
                noun_labels = self.v_labels[valid_indices, 1]
                self.verb_acc = accuracy(verb_preds.float(), verb_labels)
                self.noun_acc = accuracy(noun_preds.float(), noun_labels)
                self.mt_action_acc = multitask_accuracy(
                        (verb_preds.float(), noun_preds.float()),
                        (verb_labels, noun_labels)
                    )

            action_preds = (self.action_preds[valid_indices].float() / seen.repeat(1, self.action_preds.size(1))).softmax(dim=1)
            action_labels = self.v_labels[valid_indices, 2]
            self.action_acc = accuracy(action_preds.float(), action_labels)

        if "audio" in self.modality:
            valid_indices = (self.a_labels != -1)
            seen = self.seen_count[valid_indices].unsqueeze(1)

            aud_preds = (self.aud_preds[valid_indices].float() / seen.repeat(1, self.aud_preds.size(1))).softmax(dim=1)
            aud_labels = self.a_labels[valid_indices]

            self.aud_acc = accuracy(aud_preds.float(), aud_labels)

        if self.dataset == 'ave' and self.modality == 'audio_visual':
            combined_preds = (action_preds + aud_preds) / 2.0
            self.combined_acc = accuracy(combined_preds.float(), action_labels)

    def get_train_epoch_stats(self, iters):
        stats_dict = {
                    "train_step": iters
                }

        if "visual" in self.modality:
            if self.include_verb_noun:
                stats_dict.update(
                    {
                        "Train_Epoch/visual/verb_loss": self.visual_verb_losses.avg,
                        "Train_Epoch/visual/noun_loss": self.visual_noun_losses.avg,
                        "Train_Epoch/visual/verb_Top1_acc": self.verb_acc[0],
                        "Train_Epoch/visual/verb_Top5_acc": self.verb_acc[1],
                        "Train_Epoch/visual/noun_Top1_acc": self.noun_acc[0],
                        "Train_Epoch/visual/noun_Top5_acc": self.noun_acc[1],
                        "Train_Epoch/visual/Top1_acc": self.mt_action_acc[0],
                        "Train_Epoch/visual/Top5_acc": self.mt_action_acc[1]
                    }
                )
            stats_dict.update(
                    {
                        "Train_Epoch/visual/loss": self.visual_losses.avg,
                        "Train_Epoch/visual/action_loss": self.visual_action_losses.avg,
                        "Train_Epoch/visual/act_Top1_acc": self.action_acc[0],
                        "Train_Epoch/visual/act_Top5_acc": self.action_acc[1]
                    }
                )
        if "audio" in self.modality:
            stats_dict.update(
                    {
                        "Train_Epoch/audio/loss": self.audio_losses.avg,
                        "Train_Epoch/audio/Top1_acc": self.aud_acc[0],
                        "Train_Epoch/audio/Top5_acc": self.aud_acc[1]
                    }
                )
        
        if self.dataset == 'ave' and self.modality == 'audio_visual':
            stats_dict.update(
                    {
                        "Train_Epoch/combined/Top1_acc": self.combined_acc[0],
                        "Train_Epoch/combined/Top5_acc": self.combined_acc[1]
                    }
                )

        return stats_dict

    def get_train_epoch_message(self, epoch):
        message_str = (f'\nEpoch {epoch+1} Results:\n' \
                '\t==========================================\n')

        if "visual" in self.modality:

            message_str += (f'\tVisual Loss {self.visual_losses.avg:.5f}\n' \
                            '\t==========================================\n')
            if self.include_verb_noun:
                message_str += (f'\tVisual Views Seen: {self.visual_losses.count}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Verb Acc@1 {self.verb_acc[0]:.3f}\n' \
                    f'\tVisual Verb Acc@5 {self.verb_acc[1]:.3f}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Noun Acc@1 {self.noun_acc[0]:.3f}\n' \
                    f'\tVisual Noun Acc@5 {self.noun_acc[1]:.3f}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Action Acc@1 {self.action_acc[0]:.3f}\n' \
                    f'\tVisual Action Acc@5 {self.action_acc[1]:.3f}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Acc@1 {self.mt_action_acc[0]:.3f}\n' \
                    f'\tVisual Acc@5 {self.mt_action_acc[1]:.3f}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Loss {self.visual_losses.avg:.5f}\n' \
                    '\t==========================================\n')
            else:
                message_str += (f'\tVisual Views Seen: {self.visual_losses.count}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Action Acc@1 {self.action_acc[0]:.3f}\n' \
                    f'\tVisual Action Acc@5 {self.action_acc[1]:.3f}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Loss {self.visual_losses.avg:.5f}\n' \
                    '\t==========================================\n')
        if "audio" in self.modality:
            message_str += (f'\tAudio Views Seen: {self.audio_losses.count}\n' \
                '\t------------------------------------------\n' \
                f'\tAudio Acc@1 {self.aud_acc[0]:.3f} \n' \
                f'\tAudio Acc@5 {self.aud_acc[1]:.3f}\n' \
                '\t------------------------------------------\n' \
                f'\tAudio Loss {self.audio_losses.avg:.5f}\n' \
                '\t==========================================\n')

            message_str += (f'\tAudio Loss {self.audio_losses.avg:.5f}\n' \
                            '\t==========================================\n')
        if self.dataset == 'ave' and self.modality == 'audio_visual':
            message_str += (f'\tCombined Acc@1 {self.combined_acc[0]:.3f} \n' \
                    f'\tCombined Acc@5 {self.combined_acc[1]:.3f}\n' \
                    '\t==========================================\n')

        message_str += (f'\tActions Seen: {(self.seen_count > 0).sum()}\n' \
                '\t==========================================\n')

        if self.include_dr_loc:
            message_str += f'\tDR Loc Loss {self.drloc_losses.avg:.5f}\n'

        message_str += (f'\tLoss {self.losses.avg:.5f}\n' \
                        '\t==========================================')

        return message_str

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class InferenceMeter(object):
    """Tracks multiple metrics for TIM model during validation"""
    def __init__(self, args, num_actions):

        self.best_vis_acc1 = 0
        self.best_aud_acc1 = 0
        self.best_mt_vis_acc1 = 0
        self.best_combined_acc1 = 0

        self.early_stop_period = args.early_stop_period
        self.last_best_epoch = -1

        self.verb_acc = (0.0, 0.0)
        self.noun_acc = (0.0, 0.0)
        self.mt_action_acc = (0.0, 0.0)

        self.action_acc = (0.0, 0.0)
        self.aud_acc = (0.0, 0.0)

        self.combined_acc = (0.0, 0.0)

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.losses = AverageMeter()

        self.visual_verb_losses = AverageMeter()
        self.visual_noun_losses = AverageMeter()
        self.visual_action_losses = AverageMeter()
        self.visual_losses = AverageMeter()
        self.audio_losses = AverageMeter()

        self.dataset = args.dataset
        self.modality = args.data_modality
        self.include_verb_noun = args.include_verb_noun

        # Initialize tensors.
        if self.include_verb_noun:
            self.verb_preds = torch.zeros((num_actions, args.num_class[0][0]), dtype=torch.float32)
            self.noun_preds = torch.zeros((num_actions, args.num_class[0][1]), dtype=torch.float32)
            self.action_preds = torch.zeros((num_actions, args.num_class[0][2]), dtype=torch.float32)
        else:
            self.action_preds = torch.zeros((num_actions, args.num_class[0]), dtype=torch.float32)

        self.aud_preds = torch.zeros((num_actions, args.num_class[1]), dtype=torch.float32)
        self.seen_count = torch.zeros((num_actions,), dtype=torch.float32)

        self.v_labels = torch.full(size=(num_actions, 3), fill_value=-1, dtype=torch.int32)
        self.a_labels = torch.full(size=(num_actions,), fill_value=-1, dtype=torch.int32)

        self.reset()

    def reset(self):
        self.losses.reset()

        self.visual_verb_losses.reset()
        self.visual_noun_losses.reset()
        self.visual_action_losses.reset()
        self.visual_losses.reset()
        self.audio_losses.reset()

        if self.include_verb_noun:
            self.verb_preds.zero_()
            self.noun_preds.zero_()
        self.action_preds.zero_()
        self.aud_preds.zero_()
        self.seen_count.zero_()

        self.verb_acc = (0.0, 0.0)
        self.noun_acc = (0.0, 0.0)
        self.action_acc = (0.0, 0.0)
        self.aud_acc = (0.0, 0.0)

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
            verb_preds,
            noun_preds,
            action_preds,
            aud_preds,
            v_action_ids,
            a_action_ids,
            v_labels,
            a_labels,
            visual_loss,
            visual_loss_verb,
            visual_loss_noun,
            visual_loss_action,
            audio_loss,
            valid_visual,
            valid_audio
        ):
            if valid_visual > 0 and "visual" in self.modality:
                # Update visual metrics
                self.visual_losses.update(visual_loss, valid_visual)

                if self.include_verb_noun:
                    self.visual_verb_losses.update(visual_loss_verb, valid_visual)
                    self.visual_noun_losses.update(visual_loss_noun, valid_visual)
                    self.verb_preds.index_add_(0, v_action_ids, verb_preds)
                    self.noun_preds.index_add_(0, v_action_ids, noun_preds)

                self.visual_action_losses.update(visual_loss_action, valid_visual)
                self.action_preds.index_add_(dim=0, index=v_action_ids, source=action_preds)
                self.seen_count.index_add_(dim=0, index=v_action_ids, source=torch.ones_like(v_action_ids).float())

                self.v_labels[v_action_ids] = v_labels.int()

            if valid_audio > 0 and "audio" in self.modality:
                # Update audio metrics
                self.audio_losses.update(audio_loss, valid_audio)

                self.aud_preds.index_add_(0, a_action_ids, aud_preds)
                self.seen_count.index_add_(0, a_action_ids, torch.ones_like(a_action_ids).float())

                self.a_labels[a_action_ids] = a_labels.int()

    def update_epoch(self, epoch):
        if "visual" in self.modality:
            valid_indices = (self.v_labels[:, 2] != -1)
            seen = self.seen_count[valid_indices].unsqueeze(1)

            if self.include_verb_noun:
                verb_preds = (self.verb_preds[valid_indices] / seen.repeat(1, self.verb_preds.size(1))).softmax(dim=1)
                noun_preds = (self.noun_preds[valid_indices] / seen.repeat(1, self.noun_preds.size(1))).softmax(dim=1)
                verb_labels = self.v_labels[valid_indices, 0]
                noun_labels = self.v_labels[valid_indices, 1]
                self.verb_acc = accuracy(verb_preds, verb_labels)
                self.noun_acc = accuracy(noun_preds, noun_labels)
                self.mt_action_acc = multitask_accuracy(
                        (verb_preds, noun_preds),
                        (verb_labels, noun_labels)
                    )

            action_preds = (self.action_preds[valid_indices] / seen.repeat(1, self.action_preds.size(1))).softmax(dim=1)
            action_labels = self.v_labels[valid_indices, 2]
            self.action_acc = accuracy(action_preds, action_labels)

        if "audio" in self.modality:
            valid_indices = (self.a_labels != -1)
            seen = self.seen_count[valid_indices].unsqueeze(1)

            aud_preds = (self.aud_preds[valid_indices] / seen.repeat(1, self.aud_preds.size(1))).softmax(dim=1)
            aud_labels = self.a_labels[valid_indices]

            self.aud_acc = accuracy(aud_preds, aud_labels)

        if self.dataset == 'ave' and self.modality == 'audio_visual':
            combined_preds = (action_preds + aud_preds) / 2.0
            self.combined_acc = accuracy(combined_preds, action_labels)

        is_best_visual = self.action_acc[0] > self.best_vis_acc1
        is_best_mt_visual = self.mt_action_acc[0] > self.best_mt_vis_acc1
        is_best_audio = self.aud_acc[0] > self.best_aud_acc1
        is_best_combined = (self.combined_acc[0] > self.best_combined_acc1) & (self.dataset == 'ave' )

        best_acc1 = {
                        "visual": self.best_vis_acc1,
                        "visual_mt": self.best_mt_vis_acc1,
                        "audio": self.best_aud_acc1,
                        "combined": self.best_combined_acc1
                    }

        is_best = "none"
        if is_best_visual:
            is_best = "act_visual" if is_best_visual else ""
            self.best_vis_acc1 = max(self.action_acc[0], self.best_vis_acc1)
            self.last_best_epoch = epoch
        if is_best_mt_visual:
            is_best = "mt_visual" if is_best == "" else is_best + "_mt_visual"
            self.best_mt_vis_acc1 = max(self.mt_action_acc[0], self.best_mt_vis_acc1)
        if is_best_audio:
            is_best = "audio_" + is_best if "visual" in is_best else "audio"
            self.best_aud_acc1 = max(self.aud_acc[0], self.best_aud_acc1)
        if self.dataset == 'ave' and is_best_combined:
            is_best = "combined_" + is_best if is_best != "none" else "combined"
            self.best_combined_acc1 = self.combined_acc[0]

        if self.early_stop_period > 0:
            stop = (epoch - self.last_best_epoch) > self.early_stop_period
        else:
            stop = False

        return best_acc1, is_best, stop

    def get_val_message(self, epoch, i, dataloader_size):
        message_str = ('| Epoch: [{0}][{1}/{2}] |'
                    ' Time: {batch_time:.3f} |'
                    ' Data: {data_time:.3f} |'
                    ' Net: {net_time:.3f} |'.format(
                                        epoch+1,
                                        i+1,
                                        dataloader_size,
                                        batch_time=self.iter_timer.seconds(),
                                        data_time=self.data_timer.seconds(),
                                        net_time=self.net_timer.seconds()
                                    )
                    )
        if "visual" in self.modality:
            message_str += (' Visual Views Seen: {visual_loss.count} |'
                            ' Visual Loss: {visual_loss.avg:.4f} |'.format(
                                    visual_loss=self.visual_losses
                                )
                        )
        if "audio" in self.modality:
            message_str += (' Audio Views Seen: {audio_loss.count} |'
                            ' Audio Loss: {audio_loss.avg:.4f} |'.format(
                                    audio_loss=self.audio_losses
                                )
                        )

        message_str += (' RAM: {ram[0]:.2f}/{ram[1]:.2f}GB |'
                        ' GPU: {gpu[0]:.2f}/{gpu[1]:.2f}GB |'.format(
                                                loss=self.losses,
                                                ram=misc.cpu_mem_usage(),
                                                gpu=misc.gpu_mem_usage()
                                            )
                        )
        return message_str

    def get_val_epoch_message(self, epoch):
        message_str =(f'\nEpoch {epoch+1} Results:\n' \
                '\t==========================================\n')
        if "visual" in self.modality:
            if self.include_verb_noun:
                message_str += (f'\tVisual Views Seen: {self.visual_losses.count}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Verb Acc@1 {self.verb_acc[0]:.3f}\n' \
                    f'\tVisual Verb Acc@5 {self.verb_acc[1]:.3f}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Noun Acc@1 {self.noun_acc[0]:.3f}\n' \
                    f'\tVisual Noun Acc@5 {self.noun_acc[1]:.3f}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Action Acc@1 {self.action_acc[0]:.3f}\n' \
                    f'\tVisual Action Acc@5 {self.action_acc[1]:.3f}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Acc@1 {self.mt_action_acc[0]:.3f}\n' \
                    f'\tVisual Acc@5 {self.mt_action_acc[1]:.3f}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Loss {self.visual_losses.avg:.5f}\n' \
                    '\t==========================================\n')
            else:
                message_str += (f'\tVisual Views Seen: {self.visual_losses.count}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Action Acc@1 {self.action_acc[0]:.3f}\n' \
                    f'\tVisual Action Acc@5 {self.action_acc[1]:.3f}\n' \
                    '\t------------------------------------------\n' \
                    f'\tVisual Loss {self.visual_losses.avg:.5f}\n' \
                    '\t==========================================\n')
        if "audio" in self.modality:
            message_str += (f'\tAudio Views Seen: {self.audio_losses.count}\n' \
                '\t------------------------------------------\n' \
                f'\tAudio Acc@1 {self.aud_acc[0]:.3f} \n' \
                f'\tAudio Acc@5 {self.aud_acc[1]:.3f}\n' \
                '\t------------------------------------------\n' \
                f'\tAudio Loss {self.audio_losses.avg:.5f}\n' \
                '\t==========================================\n')
        if self.dataset == 'ave' and self.modality == 'audio_visual':
            message_str += (f'\tCombined Acc@1 {self.combined_acc[0]:.3f} \n' \
                    f'\tCombined Acc@5 {self.combined_acc[1]:.3f}\n' \
                    '\t==========================================\n')

        message_str += (f'\tActions Seen: {(self.seen_count != 0).sum()}\n' \
                '\t==========================================')


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
                        "Val/visual/noun_loss": self.visual_noun_losses.avg,
                        "Val/visual/verb_Top1_acc": self.verb_acc[0],
                        "Val/visual/verb_Top5_acc": self.verb_acc[1],
                        "Val/visual/noun_Top1_acc": self.noun_acc[0],
                        "Val/visual/noun_Top5_acc": self.noun_acc[1],
                        "Val/visual/Top1_acc": self.mt_action_acc[0],
                        "Val/visual/Top5_acc": self.mt_action_acc[1]
                    }
                )
            stats_dict.update(
                    {
                        "Val/visual/loss": self.visual_losses.avg,
                        "Val/visual/action_loss": self.visual_action_losses.avg,
                        "Val/visual/act_Top1_acc": self.action_acc[0],
                        "Val/visual/act_Top5_acc": self.action_acc[1],
                        "Val/max_vis_top1_acc": self.best_vis_acc1,
                    }
                )
        if "audio" in self.modality:
            stats_dict.update(
                    {
                        "Val/audio/loss": self.audio_losses.avg,
                        "Val/audio/Top1_acc": self.aud_acc[0],
                        "Val/audio/Top5_acc": self.aud_acc[1],
                        "Val/max_aud_top1_acc": self.best_aud_acc1,
                    }
                )
        
        if self.dataset == 'ave' and self.modality == 'audio_visual':
            stats_dict.update(
                    {
                        "Val/combined/Top1_acc": self.combined_acc[0],
                        "Val/combined/Top5_acc": self.combined_acc[1],
                        "Val/max_combined_top1_acc": self.best_combined_acc1,
                    }
                )

        return stats_dict

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class FeatureMeter(object):
    """Tracks multiple metrics for TIM during validation"""
    def __init__(
            self,
            num_actions,
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

        # Initialize tensors.
        if self.include_verb_noun:
            self.verb_preds = torch.zeros((num_actions, args.num_class[0][0]))
            self.noun_preds = torch.zeros((num_actions, args.num_class[0][1]))
            self.action_preds = torch.zeros((num_actions, args.num_class[0][2]))
        else:
            self.action_preds = torch.zeros((num_actions, args.num_class[0]))

        self.aud_preds = torch.zeros((num_actions, args.num_class[1]))
        self.seen_count = torch.zeros((num_actions,), dtype=torch.float32)
        self.narration_ids = np.zeros(num_actions, dtype=object)
        self.last_visual = 0


        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        if self.include_verb_noun:
            self.verb_preds.zero_()
            self.noun_preds.zero_()

        self.action_preds.zero_()
        self.aud_preds.zero_()
        self.seen_count.zero_()
        self.narration_ids.fill(0)

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


    def update(self, features, metadata):
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
            n_ids = metadata['v_narration_ids']
            visual_indices = np.where(np.core.defchararray.find(n_ids,'v_')!=-1)
            v_action_ids = metadata['v_action_ids'].cpu()
            v_action_ids = v_action_ids[visual_indices]
            if v_action_ids.shape[0] > 0:
                self.last_visual = max(self.last_visual, max(v_action_ids) + 1)

                if self.include_verb_noun:
                    verb_preds = features[0].cpu()
                    noun_preds = features[1].cpu()
                    self.verb_preds.index_add_(0, v_action_ids, verb_preds[visual_indices])
                    self.noun_preds.index_add_(0, v_action_ids, noun_preds[visual_indices])

                action_preds = features[2].cpu()

                self.action_preds.index_add_(0, v_action_ids, action_preds[visual_indices])
                self.seen_count.index_add_(0, v_action_ids, torch.ones_like(v_action_ids).float())

                self.narration_ids[v_action_ids] = n_ids[visual_indices]

        if "audio" in self.modality:
            n_ids = metadata['a_narration_ids']
            aud_preds = features[3].cpu()

            audio_indices = np.where(np.core.defchararray.find(n_ids,'a_')!=-1)
            a_action_ids = metadata['a_action_ids'].cpu()
            a_action_ids = a_action_ids[audio_indices]
            if a_action_ids.shape[0] > 0:
                self.aud_preds.index_add_(0, a_action_ids, aud_preds[audio_indices])
                self.seen_count.index_add_(0, a_action_ids, torch.ones_like(a_action_ids).float())

                self.narration_ids[a_action_ids] = n_ids[audio_indices]

    def get_feat_message(self, iters, total_iters):
        return ('| [{0}/{1}] | Features Extracted: {num_feats} |'
                ' Time: {batch_time:.3f} |'
                ' Data: {data_time:.3f} |'
                ' Net: {net_time:.3f} |'
                ' RAM: {ram[0]:.2f}/{ram[1]:.2f} GB |'
                ' GPU: {gpu[0]:.2f}/{gpu[1]:.2f} GB |'.format(
                                            iters+1,
                                            total_iters,
                                            num_feats=(self.seen_count > 0).sum(),
                                            batch_time=self.iter_timer.seconds(),
                                            data_time=self.data_timer.seconds(),
                                            net_time=self.net_timer.seconds(),
                                            ram=misc.cpu_mem_usage(),
                                            gpu=misc.gpu_mem_usage()
                                        )
            )

    def finalize_metrics(self):
        missing = torch.where(self.seen_count == 0)[0]
        assert  missing.size(0) == 0, f"Actions Missed: {missing}"

        if "visual" in self.modality:
            visual_seen = self.seen_count[:self.last_visual].unsqueeze(1)
            if self.include_verb_noun:
                self.verb_preds = (self.verb_preds[:self.last_visual] / visual_seen.repeat(1, self.verb_preds.size(1))).softmax(dim=1)
                self.noun_preds = (self.noun_preds[:self.last_visual] / visual_seen.repeat(1, self.noun_preds.size(1))).softmax(dim=1)

            self.action_preds = (self.action_preds[:self.last_visual] / visual_seen.repeat(1, self.action_preds.size(1))).softmax(dim=1)

        if "audio" in self.modality:
            audio_seen = self.seen_count[self.last_visual:].unsqueeze(1)
            self.aud_preds = (self.aud_preds[self.last_visual:] / audio_seen.repeat(1, self.aud_preds.size(1))).softmax(dim=1)


        data = {
                    "action": self.action_preds.numpy(),
                    "audio": self.aud_preds.numpy(),
                    "v_narration_ids": self.narration_ids[:self.last_visual],
                    "a_narration_ids": self.narration_ids[self.last_visual:]
                }
        if self.include_verb_noun:
            data.update(
                    {
                        "verb": self.verb_preds.numpy(),
                        "noun": self.noun_preds.numpy()
                    }
                )
        time_taken = str(datetime.timedelta(seconds=int(self.total_time)))
        final_message = (f'| Features Extracted: {(self.seen_count > 0).sum()} |'
                         f' Time Elapsed: {time_taken} |'
                         f' Peak RAM: {self.peak_cpu_mem:.2f} GB |'
                         f' Peak GPU: {self.peak_gpu_mem:.2f} GB |'
                    )
        logger.info(final_message)
        return data
