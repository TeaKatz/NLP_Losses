import torch

from torch.nn.functional import cosine_similarity
from torch.nn import L1Loss, MSELoss, HuberLoss, CrossEntropyLoss, BCEWithLogitsLoss, TripletMarginWithDistanceLoss


def cal_mae_loss(preds, targets, reduction="mean", **kwargs):
    """
    preds: (batch_size, )
    targets: (batch_size, )
    """
    return L1Loss(reduction=reduction)(preds, targets)


def cal_mse_loss(preds, targets, reduction="mean", **kwargs):
    """
    preds: (batch_size, )
    targets: (batch_size, )
    """
    return MSELoss(reduction=reduction)(preds, targets)


def cal_huber_loss(preds, targets, reduction="mean", **kwargs):
    """
    preds: (batch_size, )
    targets: (batch_size, )
    """
    return HuberLoss(reduction=reduction)(preds, targets)


def cal_categorical_crossentropy_loss(preds, targets, reduction="mean", ignore_index=-100, **kwargs):
    """
    preds: (batch_size, class_size)
    targets: (batch_size, )
    """
    return CrossEntropyLoss(reduction=reduction, 
                            ignore_index=ignore_index)(preds, targets)


def cal_binary_crossentropy_loss(preds, targets, reduction="mean", ignore_index=None, **kwargs):
    """
    preds: (batch_size, class_size)
    targets: (batch_size, class_size)
    """
    mask = torch.ones_like(preds)
    if ignore_index is not None:
        assert ignore_index >= 0, "ignore_index cannot < 0"
        mask[:, ignore_index] = 0.

    # (batch_size, class_size)
    loss = BCEWithLogitsLoss(reduction="none")(preds, targets)
    loss = loss * mask
    if reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "mean":
        loss = torch.mean(loss)
    return loss


def cal_cosine_similarity_loss(preds1, preds2, targets, margin=0.0, reduction="mean", **kwargs):
    """
    preds1: (batch_size, vector_size)
    preds2: (batch_size, vector_size)
    targets: (batch_size, )
    """

    loss = torch.maximum(0, torch.abs(cosine_similarity(preds1, preds2) - targets) - margin)
    if reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "mean":
        loss = torch.mean(loss)
    return loss


def cal_logistic_loss(preds, targets, reduction="mean", **kwargs):
    """
    preds: (batch_size, vector_size)
    targets: (batch_size, vector_size)
    """
    # (batch_size, )
    scalar_product = torch.matmul(preds, targets.transpose(0, 1)).diagonal()
    loss = torch.log(1 + torch.exp(-scalar_product))
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    return loss


def cal_negative_logistic_loss(preds, targets, reduction="mean", **kwargs):
    """
    preds: (batch_size, vector_size)
    targets: (batch_size, vector_size)
    """
    # (batch_size, )
    scalar_product = torch.matmul(preds, targets.transpose(0, 1)).diagonal()
    loss = torch.log(1 + torch.exp(scalar_product))
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    return loss


def cal_fasttext_loss(anchors, positives, negatives, reduction="mean", **kwargs):
    """
    anchors: (batch_size, vector_size)
    positives: (batch_size, vector_size)
    negatives: (batch_size, negative_size, vector_size)
    """
    _, negative_size, _ = negatives.shape

    pos_loss = cal_logistic_loss(anchors, positives, reduction=reduction)
    neg_loss = 0.0
    for i in range(negative_size):
        neg_loss += cal_negative_logistic_loss(anchors, negatives[:, i], reduction=reduction)
    return pos_loss + neg_loss


def cal_triplet_loss(anchors, positives, negatives, distance_function=None, margin=0.0, reduction="mean", **kwargs):
    return TripletMarginWithDistanceLoss(distance_function=distance_function, 
                                         margin=margin, 
                                         reduction=reduction)(anchors, positives, negatives)


class Losses:
    name2loss = {
        "MAELoss": cal_mae_loss,
        "MSELoss": cal_mse_loss,
        "HuberLoss": cal_huber_loss,
        "CategoricalCrossEntropyLoss": cal_categorical_crossentropy_loss,
        "BinaryCrossEntropyLoss": cal_binary_crossentropy_loss,
        "CosineSimilarityLoss": cal_cosine_similarity_loss,
        "LogisticLoss": cal_logistic_loss,
        "NegativeLogisticLoss": cal_negative_logistic_loss,
        "FastTextLoss": cal_fasttext_loss,
        "TripletLoss": cal_triplet_loss
    }
    def __init__(self, losses , **losses_arguments):
        if not isinstance(losses, list):
            losses = [losses]
        self.losses = [self.name2loss[name] for name in losses]
        self.losses_arguments = losses_arguments

    def __call__(self, *args):
        loss = 0.
        for loss_func in self.losses:
            loss += loss_func(*args, **self.losses_arguments)
        return loss
