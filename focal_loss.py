import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        :param gamma: focusing parameter (typically 1–3)
        :param alpha: class weights (tensor of shape [num_classes]) or float
                      used to reduce class imbalance
        :param reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif alpha is not None:
            # scalar alpha (for binary imbalance)
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)

    def forward(self, logits, targets):
        """
        logits: predictions from model (batch_size, num_classes)
        targets: ground-truth labels (batch_size)
        """
        # Compute log-probabilities
        log_probs = F.log_softmax(logits, dim=1)
        
        # Probabilities
        probs = torch.exp(log_probs)

        # Select the probabilities corresponding to the true class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal loss components
        focal_term = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction
