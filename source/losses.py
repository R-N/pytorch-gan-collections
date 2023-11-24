import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogits(nn.BCEWithLogitsLoss):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = super().forward(pred_real, torch.ones_like(pred_real))
            loss_fake = super().forward(pred_fake, torch.zeros_like(pred_fake))
            return loss_real + loss_fake
        else:
            loss = super().forward(pred_real, torch.ones_like(pred_real))
            return loss


class Hinge(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            return loss_real + loss_fake
        else:
            loss = -pred_real.mean()
            return loss


class Wasserstein(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = -pred_real.mean()
            loss_fake = pred_fake.mean()
            loss = loss_real + loss_fake
            return loss
        else:
            loss = -pred_real.mean()
            return loss


class Softplus(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.softplus(-pred_real).mean()
            loss_fake = F.softplus(pred_fake).mean()
            loss = loss_real + loss_fake
            return loss
        else:
            loss = F.softplus(-pred_real).mean()
            return loss

def reduce(loss, reduction="mean"):
    if reduction and reduction != "none":
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    return loss

def mse(pred, target):
    return ((pred - target) ** 2).mean()
# THIS IS NOT STANDARD MSLE
def mile(pred, y, reduction="mean"):
    error = torch.abs(pred - y)
    loss = mile_(error)
    loss = reduce(loss, reduction=reduction)
    return loss

def mile_(error):
    return torch.log(1+error) * (1+error) - error

def mire(pred, y, scale=0.5, reduction="mean"):
    error = torch.abs(pred - y)
    loss = mire_(error, scale=scale)
    loss = reduce(loss, reduction=reduction)
    return loss

def mire_(error, scale=0.5):
    return scale * torch.pow(error, 1.5)
