import torch
from torch.nn import functional as F

from optimizers.darts.architect import Architect


class ArchitectConsistency(Architect):

    def __init__(self, model, args):
        super(ArchitectConsistency, self).__init__(model, args)
        self.sesu_loss_weighting = args.sesu_loss_weighting
        self.sup_loss_weighting = args.sup_loss_weighting

    def _val_loss(self, model, input, target):
        labelled_inp, unlabelled_augm_inp, unlabelled_non_augm_inp = input
        labelled_tar, _, _ = target

        supervised_loss = model._loss(labelled_inp, labelled_tar)

        pred_sesu = model(unlabelled_non_augm_inp)
        pred_sesu_augm = model(unlabelled_augm_inp)
        sesu_loss = self._kl_divergence_with_logits(p_logits=pred_sesu.detach(), q_logits=pred_sesu_augm).mean()
        loss = supervised_loss * self.sup_loss_weighting + sesu_loss * self.sesu_loss_weighting
        return loss

    # Semi supervised loss equivalent to:
    # https://github.com/google-research/uda/blob/master/image/main.py#L247
    def _kl_divergence_with_logits(self, p_logits, q_logits):
        p = F.softmax(p_logits, dim=-1)
        log_p = F.log_softmax(p_logits, dim=-1)
        log_q = F.log_softmax(q_logits, dim=-1)
        kl = torch.sum(p * (log_p - log_q), dim=-1)
        return kl
