import torch.nn as nn
import torch
import numpy as np

def func1(y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        _negative_weight = 0.5
        _margin = 0.6

        pos_logits = y_pred[:, 0]
        print(f"pos_logits: {pos_logits}")
        pos_loss = torch.relu(1 - pos_logits)
        print(f"pos_loss: {pos_loss}")

        neg_logits = y_pred[:, 1:]
        print(f"neg_logits: {neg_logits}")
        neg_loss = torch.relu(neg_logits - _margin)
        print(f"neg_loss: {neg_loss}")

        if _negative_weight:
            print(f"neg_loss mean: {neg_loss.mean(dim=-1)}")
            print(f"neg_loss mean * _negative_weight: {neg_loss.mean(dim=-1) * _negative_weight}")
            loss = pos_loss + neg_loss.mean(dim=-1) * _negative_weight
            print(f"loss: {loss}")
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)
        return loss.mean()

def func21(margin_values_list, weights_list, neg_logits):
      neg_loss = torch.zeros_like(neg_logits)
      for i in range(0, len(weights_list)):
        neg_loss_tmp = torch.relu(neg_logits - margin_values_list[i])
        print(f"neg_loss_tmp: {neg_loss_tmp}, margin: {margin_values_list[i]}")
        neg_loss_tmp = neg_loss_tmp * weights_list[i]
        print(f"neg_loss_tmp * _negative_weigt: {neg_loss_tmp}, _negative_weigt: {weights_list[i]}")
        neg_loss += neg_loss_tmp
      print(f"neg_loss: {neg_loss}")
      print(f"neg_loss_v1 mean: {neg_loss.mean(dim=-1)}")
      return neg_loss.mean(dim=-1)

def func22(margin_values_list, weights_list, neg_logits):
  margin_values = torch.tensor(margin_values_list).reshape(-1, 1)
  weights = torch.tensor(weights_list).reshape(-1, 1)

  print(neg_logits.shape)
  print(margin_values.shape)
  print(weights.shape)
  neg_losses = torch.relu(neg_logits - margin_values)
  print(f"neg_logits - margin_values: {neg_logits - margin_values}")
  neg_losess =neg_losses * weights
  print(f"neg_losses: {neg_losses}")
  neg_loss = torch.sum(neg_losses, dim=0)
  print(f"neg_loss: {neg_loss}")
  print(f"neg_los mean: {neg_loss.mean(dim=-1)}")
  return neg_loss.mean(dim=-1)

def func23(margin_values_list, weights_list, neg_logits):
    margin_values = torch.tensor(margin_values_list)
    weights = torch.tensor(weights_list)
    neg_logits_prime =  neg_logits.reshape(-1, 1)
    print(f"neg_logits_prime - margin_values: {neg_logits_prime - margin_values}")
    neg_losses = torch.relu(neg_logits_prime - margin_values) * weights
    print(f"neg_losses: {neg_losses}")
    neg_loss = torch.sum(neg_losses, dim=-1)
    print(f"neg_loss: {neg_loss}")
    print(f"neg_los mean: {neg_loss.mean(dim=-1)}")
    return neg_loss.mean(dim=-1)

def func24(margin_values_list, weights_list, neg_logits):
    margin_values = torch.tensor(margin_values_list).unsqueeze(-1)
    weights = torch.tensor(weights_list).unsqueeze(-1)#.view(1, -1)

    # print(margin_values)
    # print(neg_logits)
    # print(neg_logits.unsqueeze(1))
    # print(neg_logits.shape)
    # print(neg_logits.unsqueeze(1).shape)

    neg_loss_tmp = torch.relu(neg_logits.unsqueeze(1) - margin_values)
    print(f"neg_logits.unsqueeze(1) - margin_values: {neg_logits.unsqueeze(1) - margin_values}")
    print(f"neg_loss_tmp: {neg_loss_tmp}")

    neg_loss_tmp = neg_loss_tmp * weights
    print(f"neg_loss_tmp * _negative_weigt: {neg_loss_tmp}")

    neg_loss = torch.sum(neg_loss_tmp, dim=1)
    print(f"neg_loss: {neg_loss}")
    print(f"neg_los mean: {neg_loss.mean(dim=-1)}")
    return neg_loss.mean(dim=-1)



def func2(y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        _margin1 = 0.1 # easy
        _margin2 = 0.3 # mid
        _margin3 = 0.5 # hard
        _margin4 = 0.6 # hardest
        _negative_weight1 = 0.1 # easy, totalweight = 0.1
        _negative_weight2 = 0.4 # mid, totalweight = 0.5
        _negative_weight3 = 0.6 # hard, totalweight = 1.1
        _negative_weight4 = 0.9 # hardest, totalweight = 2.0


        pos_logits = y_pred[:, 0]
        print(f"pos_logits: {pos_logits}")
        pos_loss = torch.relu(1 - pos_logits)
        print(f"pos_loss: {pos_loss}")

        neg_logits = y_pred[:, 1:]
        print(f"neg_logits: {neg_logits}")

        margins = [_margin1, _margin2, _margin3, _margin4]
        weights = [_negative_weight1, _negative_weight2, _negative_weight3, _negative_weight4]
        print("---v1------")
        neg_loss_mean = func21(margins, weights,neg_logits)
        # print("---v2------")
        # neg_loss_mean = func22(margins, weights, neg_logits)
        # print("--v3-------")
        # neg_loss_mean = func23(margins, weights,neg_logits)
        print("--v4-------")
        neg_loss_mean = func24(margins, weights,neg_logits)
        print("---------")


        loss = pos_loss + neg_loss_mean
        print(f"loss: {loss}")

        return loss.mean()

import torch

def calculate_neg_loss(neg_logits, margin_values_list, weights_list):
    margin_values = torch.tensor(margin_values_list).unsqueeze(-1) #.view(1, -1)
    weights = torch.tensor(weights_list).unsqueeze(-1)#.view(1, -1)

    # print(margin_values.shape)
    # print(margin_values)

    # print(neg_logits.shape)
    # print(neg_logits)
    # print(neg_logits.unsqueeze(1).shape)
    # print(neg_logits.unsqueeze(1))

    neg_losses = torch.relu(neg_logits[:, None, :] - margin_values) * weights
    neg_loss = neg_losses.sum(dim=1).mean(dim=-1)



    print(f"neg_logits.unsqueeze(1) - margin_values: {neg_logits.unsqueeze(1) - margin_values}")

    neg_loss_tmp = torch.relu(neg_logits.unsqueeze(1) - margin_values)
    print(f"neg_loss_tmp: {neg_loss_tmp}, margin: {margin_values}")

    neg_loss_tmp = neg_loss_tmp * weights
    print(f"neg_loss_tmp * _negative_weigt: {neg_loss_tmp}, _negative_weigt: {weights}")

    neg_lossv1 = torch.sum(neg_loss_tmp, dim=1)
    print(f"neg_lossv1: {neg_lossv1}")
    print(f"neg_lossv1 mean: {neg_lossv1.mean(dim=1)}")

    return neg_loss

# Example usage:
neg_logits = y_pred[:, 1:] # torch.randn((2, 5))
margin_values_list = [0.1, 0.2, 0.3, 0.4]
weights_list = [0.5, 0.3, 0.2, 0.1]

result = calculate_neg_loss(neg_logits, margin_values_list, weights_list)
print(result)


# import numpy as np
# import torch

# def f(y_pred, y_true, pos_weight, neg_weight, margin_values, negative_weights):
#   """
#   :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
#   :param y_true: true labels of shape (batch_size, 1 + num_negs)
#   """
#   pos_logits = y_pred[:, 0]
#   print(f"pos_logits: {pos_logits}")

#   # pos_loss = torch.relu(1 - pos_logits)
#   # print(f"pos_loss: {pos_loss}")

#   print(f"-pos_logits+margin_values: {-pos_logits +margin_values}")

#   pos_losses = torch.relu(-pos_logits + margin_values) * negative_weights
#   print(f"pos_losses: {pos_losses}")
#   pos_loss_mean = pos_losses.sum(dim=1).mean(dim=-1)
#   print(f"pos_loss_mean: {pos_loss_mean}")

#   neg_logits = y_pred[:, 1:]
#   neg_losses = torch.relu(neg_logits[:, None, :] - margin_values) * negative_weights
#   print(f"neg_lossesv1: {neg_losses}")

#   # mask = torch.relu(neg_logits[:, None, :] - margin_values) != 0
#   # neg_losses = mask * neg_logits[:, None, :] * negative_weights
#   # print(f"neg_losses: {neg_losses}")

#   neg_loss_mean = neg_losses.sum(dim=1).mean(dim=-1)

#   # print(f"neg_losses.sum(dim=1):{neg_losses.sum(dim=1)}")
#   # print(f"neg_losses.mean(dim=1):{neg_losses.mean(dim=1)}")
#   # print(f"neg_losses.mean(dim=1).mean(dim=-1):{neg_losses.mean(dim=1).mean(dim=-1)}")

#   print(f"neg_loss_mean: {neg_loss_mean}")

#   # mask = neg_losses > 0
#   # neg_loss_mean2 = (neg_losses*mask).sum(dim=1)/mask.sum(dim=1)
#   # print(f"neg_loss_mean2: {neg_loss_mean2}")
#   # print(f"neg_loss_mean2..mean(dim=-1): {neg_loss_mean2.mean(dim=-1)}")

#   # loss = pos_weight * pos_loss + neg_weight * neg_loss_mean
#   # print(f"loss: {loss}")
#   loss = pos_weight * pos_loss_mean + neg_weight * neg_loss_mean
#   print(f"loss: {loss}")

#   # neg_loss = torch.relu(neg_logits - self._margin)
#   # if self._negative_weight:
#   #     loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
#   # else:
#   #     loss = pos_loss + neg_loss.sum(dim=-1)

#   return loss.mean()


import torch
import numpy as np


# margin_values_list= [0.1, 0.5, 0.8]
# negative_weights_list= [0.0, 0.0, 1.0]
margin_values_list= [0.8]
negative_weights_list= [1.0]
_pos_weight= 0.5
_neg_weight= 0.5

device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
_margin_values = torch.tensor(margin_values_list).unsqueeze(-1).to(device)
_negative_weights = torch.tensor(negative_weights_list).unsqueeze(-1).to(device)



def f(y_pred, y_true):
      """
      :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
      :param y_true: true labels of shape (batch_size, 1 + num_negs)
      """
      mean_version=0

      pos_logits = y_pred[:, 0]
      pos_loss = _helper_forward_pos_v0(pos_logits, mean_version)
      #pos_loss = self._helper_forward_pos_v1(pos_logits)

      neg_logits = y_pred[:, 1:]
      neg_loss = _helper_forward_neg_v0(neg_logits, mean_version)
      # neg_loss = self._helper_forward_neg_v1(neg_logits, mean_version)
      loss = _pos_weight * pos_loss + _neg_weight * neg_loss

      #org impl
      # neg_loss = torch.relu(neg_logits - self._margin)
      # if self._negative_weight:
      #     loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
      # else:
      #     loss = pos_loss + neg_loss.sum(dim=-1)
      return loss.mean()

def _helper_forward_pos_v0( pos_logits, mean_version=0):
    pos_loss = torch.relu(1 - pos_logits)
    print(f"pos_loss: {pos_loss}")
    return pos_loss

def _helper_forward_pos_v1(pos_logits, mean_version=0):
    # (1-pos) - (1-margin) = margin-pos
    # 1-neg weights: i.e., opposite of negatives
    pos_losses = torch.relu(-pos_logits + _margin_values) * (1-_negative_weights)
    pos_loss = _helper_mean(pos_losses, input_dim=0, mean_version=mean_version)
    print(f"pos_loss: {pos_loss}")
    return pos_loss

def _helper_forward_neg_v0(neg_logits, mean_version=0):
    neg_losses = torch.relu(neg_logits[:, None, :] -_margin_values) * _negative_weights
    print(f"neg_losses: {neg_losses}")
    neg_loss = _helper_mean(neg_losses, input_dim=1, mean_version=mean_version)
    print(f"neg_loss: {neg_loss}")
    return neg_loss

def _helper_forward_neg_v1(neg_logits, mean_version=0):
  mask = torch.relu(neg_logits[:, None, :] - _margin_values) != 0
  neg_losses = mask * neg_logits[:, None, :] * _negative_weights
  neg_loss =_helper_mean(neg_losses, input_dim=1, mean_version=mean_version)
  print(f"neg_loss: {neg_loss}")
  return neg_loss

def _helper_mean(losses, input_dim=1, mean_version=0):
    print(losses.shape[0])
    loss = torch.zeros(losses.shape[0])
    if mean_version == 0:
      loss = _helper_mean_v0(losses, input_dim)
    elif mean_version == 1:
      loss = _helper_mean_v1(losses, input_dim)
    elif mean_version == 2:
      loss = _helper_mean_v2(losses, input_dim)
    else:
      loss = np.nan
      print("!!!!! Wrong type of mean")
    return loss

def _helper_mean_v0(losses, input_dim=1):
  loss_mean = losses.sum(dim=input_dim).mean(dim=-1)
  return loss_mean

def _helper_mean_v1(losses, input_dim=1):
  epsilon = 1e-10
  mask = losses > 0
  print(f"losses: {losses}")
  print(f"mask: {mask}")
  print(f"losses*mask: {losses*mask}")
  print(f"(losses*mask).sum(input_dim): {(losses*mask).sum(dim=input_dim)}")
  print(f"(losses*mask).sum(input_dim)/mask.sum(input_dim): {(losses*mask).sum(dim=input_dim)/mask.sum(dim=input_dim)}")

  loss_mean_tmp = (losses*mask).sum(dim=input_dim)/(mask.sum(dim=input_dim)+epsilon)

  mask = loss_mean_tmp > 0
  print(f"loss_mean_tmp: {loss_mean_tmp}")
  print(f"loss_mean_tmp*mask: {loss_mean_tmp*mask}")

  loss_mean = (loss_mean_tmp*mask).sum(dim=input_dim)/(mask.sum(dim=input_dim)+epsilon)

  return loss_mean

def _helper_mean_v2(losses, input_dim=1):
  epsilon = 1e-10
  print(f"losses: {losses}")

  print(f"losses.sum(dim=input_dim): {losses.sum(dim=0)}")
  print(f"losses.sum(dim=input_dim).mean(dim=input_dim): {losses.sum(dim=0).mean(dim=input_dim)}")

  loss_mean_tmp = losses.sum(dim=0).mean(dim=0)

  mask = loss_mean_tmp > 0
  print(f"loss_mean_tmp: {loss_mean_tmp}")
  print(f"loss_mean_tmp*mask: {loss_mean_tmp*mask}")

  loss_mean = (loss_mean_tmp*mask).sum(dim=input_dim)/(mask.sum(dim=input_dim)+epsilon)

  return loss_mean