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
        self._mean_version = 0

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """

        pos_logits = y_pred[:, 0]
        pos_loss = self._helper_forward_pos_v0(pos_logits, self._mean_version)
        #pos_loss = self._helper_forward_pos_v1(pos_logits)

        neg_logits = y_pred[:, 1:]
        neg_loss = self._helper_forward_neg_v0(neg_logits, self._mean_version)
        # neg_loss = self._helper_forward_neg_v1(neg_logits, mean_version)


        loss = self._pos_weight * pos_loss + self._neg_weight * neg_loss
        # neg_loss = torch.relu(neg_logits - self._margin)
        # if self._negative_weight:
        #     loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        # else:
        #     loss = pos_loss + neg_loss.sum(dim=-1)
        return loss.mean()

    def _helper_forward_pos_v0(self, pos_logits, mean_version=0):
        pos_loss = torch.relu(1 - pos_logits)
        return pos_loss

    def _helper_forward_pos_v1(self, pos_logits, mean_version=0):
        # (1-pos) - (1-margin) = margin-pos
        # 1-neg weights: i.e., opposite of negatives
        pos_losses = torch.relu(-pos_logits + self._margin_values) * (1-self._negative_weights)
        pos_loss = self._helper_mean(pos_losses, input_dim=0, mean_version=mean_version)
        return pos_loss

    def _helper_forward_neg_v0(self, neg_logits, mean_version=0):
        neg_losses = torch.relu(neg_logits[:, None, :] - self._margin_values) * self._negative_weights
        neg_loss = self._helper_mean(neg_losses, input_dim=1, mean_version=mean_version)
        return neg_loss

    def _helper_forward_neg_v1(self, neg_logits, mean_version=0):
      mask = torch.relu(neg_logits[:, None, :] - self._margin_values) != 0
      neg_losses = mask * neg_logits[:, None, :] * self._negative_weights
      neg_loss = self._helper_mean(neg_losses, input_dim=1, mean_version=mean_version)
      return neg_loss

    def _helper_mean(self, losses, input_dim=1, mean_version=0):
        loss = torch.zeros(losses.shape[0]).to(self._device)
        if mean_version == 0:
          loss = self._helper_mean_v0(losses, input_dim)
        elif mean_version == 1:
          loss = self._helper_mean_v1(losses, input_dim)
        else:
          print("!!!!! Wrong type of mean")
        return loss
    
    def _helper_mean_v0(self, losses, input_dim=1):
      loss_mean = losses.sum(dim=input_dim).mean(dim=-1)
      return loss_mean

    def _helper_mean_v1(self, losses, input_dim=1):
      # NOTE: Not sure. Do not use
      epsilon = 1e-10
      mask = losses > 0
      loss_mean_tmp = (losses*mask).sum(dim=input_dim)/(mask.sum(dim=input_dim)+epsilon)

      mask = loss_mean_tmp > 0
      loss_mean = (loss_mean_tmp*mask).sum(dim=input_dim)/(mask.sum(dim=input_dim)+epsilon)
      return loss_mean

