"""Unmodified loss and metric definitions from the original training notebook.

Only imports are added outside the verbatim cell blocks. ``label`` intentionally
comes from skimage.measure as in the final import of the original notebook.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment
from skimage.measure import label

# >>> ORIGINAL NOTEBOOK CELL 50 START (UNMODIFIED)
class Focalloss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(Focalloss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F_torch.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        if self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss
# <<< ORIGINAL NOTEBOOK CELL 50 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 52 START (UNMODIFIED)
def nIoU_Score(prediction, target, epsilon=1e-6):

    prediction_copy = prediction.clone()

    prediction_copy = torch.sigmoid(prediction_copy)
    prediction_copy = (prediction_copy > 0.5).float()

    intersection = (prediction_copy * target).sum()

    union = (prediction_copy.sum()+ target.sum()- intersection)

    iou = (intersection + epsilon) / (union + epsilon)

    return iou
# <<< ORIGINAL NOTEBOOK CELL 52 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 54 START (UNMODIFIED)
def IoU_Score(prediction, target, epsilon=1e-6):

    prediction_copy = prediction.clone()

    prediction_copy = torch.sigmoid(prediction_copy)
    prediction_copy = (prediction_copy > 0.5).float()

    intersection = (prediction_copy * target).sum()

    union = (prediction_copy.sum()+ target.sum()- intersection)

    return intersection, union
# <<< ORIGINAL NOTEBOOK CELL 54 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 55 START (UNMODIFIED)
def F_measure(prediction, mask, epsilon=1e-6):

    pred = torch.sigmoid(prediction)

    TP = (pred * mask).sum()
    FP = (pred * (1 - mask)).sum()
    FN = ((1 - pred) * mask).sum()

    eps = epsilon

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1
# <<< ORIGINAL NOTEBOOK CELL 55 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 57 START (UNMODIFIED)
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)

        intersection = (prediction * target).sum()
        union = prediction.sum() + target.sum()

        dice = ((2 * intersection) + self.eps) / (union + self.eps)

        return 1 - dice
# <<< ORIGINAL NOTEBOOK CELL 57 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 59 START (UNMODIFIED)
class DiceScore(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)

        intersection = (prediction * target).sum()
        union = prediction.sum() + target.sum()

        dice = ((2 * intersection) + self.eps) / (union + self.eps)

        return dice
# <<< ORIGINAL NOTEBOOK CELL 59 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 60 START (UNMODIFIED)
def pd_score(prediction, target, threshold=0.5, distance_threshold=3):

    pred = (torch.sigmoid(prediction) > threshold).squeeze().cpu().numpy().astype(np.uint8)
    gt = target.squeeze().cpu().numpy().astype(np.uint8)

    pred_labels, pred_num = label(pred, return_num=True)
    gt_labels, gt_num = label(gt, return_num=True)

    if pred_num == 0 or gt_num == 0:
        return 0.0

    pred_centroids = np.array(center_of_mass(pred, pred_labels, range(1, pred_num + 1)))
    gt_centroids = np.array(center_of_mass(gt, gt_labels, range(1, gt_num + 1)))

    distance_matrix = np.linalg.norm(
        gt_centroids[:, None] - pred_centroids[None, :],
        axis=2
    )

    gt_idx, pred_idx = linear_sum_assignment(distance_matrix)

    detected = np.sum(distance_matrix[gt_idx, pred_idx] < distance_threshold)

    return detected / gt_num
# <<< ORIGINAL NOTEBOOK CELL 60 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 61 START (UNMODIFIED)
class EdgeLoss(nn.Module):
    def __init__(self, bce_weight=10.0, dice_weight=1.0, eps=1e-6):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps)
        self.focal_loss = Focalloss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, prediction, target):

        #bce_loss = self.bce(prediction, target)
        dice_loss = self.dice(prediction, target)
        focal_loss = self.focal_loss(prediction, target)

        loss = (self.bce_weight * focal_loss) + dice_loss

        return loss
# <<< ORIGINAL NOTEBOOK CELL 61 END (UNMODIFIED)


# >>> ORIGINAL NOTEBOOK CELL 62 START (UNMODIFIED)
class HeteroscedasticBCELoss(nn.Module):

    def __init__(self,num_samples: int = 10,uncertainty_weight: float = 1e-4,):
        super().__init__()

        self.num_samples = num_samples
        self.uncertainty_weight = uncertainty_weight

    def forward(self,mean_logits: torch.Tensor,log_var: torch.Tensor,target: torch.Tensor):
        target = target.float()

        std = torch.exp(0.5 * log_var)

        sampled_losses = []

        for _ in range(self.num_samples):
            epsilon = torch.randn_like(mean_logits)

            noisy_logits = mean_logits + std * epsilon

            sample_loss = F_torch.binary_cross_entropy_with_logits(noisy_logits,target,reduction="mean")

            sampled_losses.append(sample_loss)

        segmentation_loss = torch.stack(sampled_losses).mean()

        # Small regularizer prevents the model from predicting
        # unnecessarily large uncertainty everywhere.
        uncertainty_regularizer = log_var.mean()

        total_loss = (segmentation_loss + self.uncertainty_weight * uncertainty_regularizer)

        return total_loss
# <<< ORIGINAL NOTEBOOK CELL 62 END (UNMODIFIED)

