import numpy as np
import torch

def dsc_score(y_pred, y_true):
    y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    score = np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))
    score = torch.tensor(score)
    
    return score