import torch.nn as nn
import torch


class MultiCosineContrastiveLoss(nn.Module):
    def __init__(
      self, pos_weight, neg_weight, 
      margin_values_list, negative_weights_list):
        """
        :param margin: float, margin in MultiCosineContrastiveLoss
        :param num_negs: int, number of negative samples
        :param negative_weight:, float, the weight set to the negative samples. When negative_weight=None, it
            equals to num_negs
        """
        super(MultiCosineContrastiveLoss, self).__init__()
        # self._margin = margin
        # self._negative_weight = negative_weight
        self._margin_values_list = margin_values_list
        self._negative_weights_list = negative_weights_list
        self._pos_weight = pos_weight
        self._neg_weight = neg_weight
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._margin_values = torch.tensor(self._margin_values_list).unsqueeze(-1).to(self._device)
        self._negative_weights = torch.tensor(self._negative_weights_list).unsqueeze(-1).to(self._device)

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = self._helper_forward_pos_v0(pos_logits)
        neg_logits = y_pred[:, 1:]
        neg_loss = self._helper_forward_neg_v0(neg_logits)
        loss = (self._pos_weight * pos_loss) + (self._neg_weight * neg_loss)
        return loss.mean()

    def _helper_forward_pos_v0(self, pos_logits):
        pos_loss = torch.relu(1 - pos_logits)
        return pos_loss

    def _helper_forward_neg_v0(self, neg_logits):
        neg_losses = torch.relu(neg_logits[:, None, :] - self._margin_values) * self._negative_weights
        neg_loss = self._helper_mean(neg_losses, input_dim=1)
        return neg_loss

    def _helper_mean(self, losses, input_dim=1):
        loss = torch.zeros(losses.shape[0]).to(self._device)
        loss = self._helper_mean_v0(losses, input_dim)
        return loss

    def _helper_mean_v0(self, losses, input_dim=1):
      loss_mean = losses.sum(dim=input_dim).mean(dim=-1)
      return loss_mean


