import torch

def mse_loss(pred, target, reduction = 'mean'):
    loss = torch.nn.MSELoss(reduction=reduction)
    return loss(pred, target)

def l1_loss(pred, target, reduction = 'mean'):
    loss = torch.nn.L1Loss(reduction=reduction)
    return loss(pred, target)
