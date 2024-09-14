# import telnetlib
# from traceback import print_tb
import torch.nn as nn
import torch
# import torch.nn.functional as F
# import numpy as np

class BSLLoss(nn.Module):
    def __init__(self, temperature=1., temperature2=1., temperature3=1., mode="test"):
        super(BSLLoss, self).__init__()
        self.temperature = temperature
        self.temperature_2 = temperature2
        self.temperature_3 = temperature3
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.mode = mode
        print("Here is Pos_DROLoss loss")
        print("tau_1\ttau_2 tau_3 \t is {}\t{}\t {}\t MODE {}".format(temperature, temperature2, temperature3, self.mode))

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs) 
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        # user = y_true
        pos_logits = torch.exp(y_pred[:, 0] / self.temperature)
        neg_logits = torch.exp(y_pred[:, 1:] / self.temperature)
        user = pos_logits
        # print(self.loss_re)
        if self.mode == "multi":
            user = user.contiguous().view(-1, 1)
            mask = torch.eq(user, user.T).float().to(self.device)
            pos_logits = (pos_logits.unsqueeze(0) * mask).sum(1) / mask.sum(1)
            neg_logits = torch.pow(torch.mean(neg_logits, dim=-1), self.temperature_2)
        elif self.mode == "single":
            pos_logits = pos_logits
            neg_logits = torch.mean(neg_logits, dim=-1)
        elif self.mode == "reweight":
            neg_logits = neg_logits.sum(dim=-1)
            neg_logits = torch.pow(neg_logits, self.temperature_2)
 
        loss = - torch.log(pos_logits / neg_logits).mean()
        return loss
