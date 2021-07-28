import torch

from torch.nn import L1Loss, MSELoss, HuberLoss, CrossEntropyLoss, BCEWithLogitsLoss, CosineEmbeddingLoss, TripletMarginWithDistanceLoss


def cal_mae_loss(inputs, targets, reduction="sum", **kwargs):
    return L1Loss(reduction=reduction)(inputs, targets)


def cal_mse_loss(inputs, targets, reduction="sum", **kwargs):
    return MSELoss(reduction=reduction)(inputs, targets)


def cal_huber_loss(inputs, targets, reduction="sum", **kwargs):
    return HuberLoss(reduction=reduction)(inputs, targets)


def cal_categorical_crossentropy_loss(inputs, targets, reduction="sum", ignore_index=-100, **kwargs):
    return CrossEntropyLoss(reduction=reduction, 
                            ignore_index=ignore_index)(inputs, targets)


def cal_binary_crossentropy_loss(inputs, targets, reduction="sum", **kwargs):
    return BCEWithLogitsLoss(reduction=reduction)(inputs, targets)


def cal_cosine_loss(inputs, targets, margin=0.0, reduction="sum", **kwargs):
    input1, input2 = inputs
    return CosineEmbeddingLoss(margin=margin, 
                               reduction=reduction)(input1, input2, targets)


def cal_logistic_loss(inputs, targets, reduction="sum", **kwargs):
    scalar_product = torch.matmul(inputs, targets.transpose(0, 1)).diagonal()
    loss = torch.log(1 - torch.exp(-scalar_product))
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    return loss


def cal_negative_logistic_loss(inputs, targets, reduction="sum", **kwargs):
    scalar_product = torch.matmul(inputs, targets.transpose(0, 1)).diagonal()
    loss = torch.log(1 - torch.exp(scalar_product))
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    return loss


def cal_triplet_loss(inputs, targets, distance_function=None, margin=0.0, reduction="sum", **kwargs):
    anchors, positives, negatives = inputs
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
        "CosineLoss": cal_cosine_loss,
        "LogisticLoss": cal_logistic_loss,
        "NegativeLogisticLoss": cal_negative_logistic_loss,
        "TripletLoss": cal_triplet_loss
    }
    def __init__(self, losses , **losses_arguments):
        if not isinstance(losses, list):
            losses = [losses]
        self.losses = [self.name2loss[name] for name in losses]
        self.losses_arguments = losses_arguments

    def __call__(self, inputs, targets):
        loss = 0.
        for loss_func in self.losses:
            loss += loss_func(inputs, targets, **self.losses_arguments)
        return loss
