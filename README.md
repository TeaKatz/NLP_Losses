# NLP_Losses
A repository gathers losses for training models.
- [MAELoss](https://github.com/TeaKatz/NLP_Losses/tree/main/src/nlp_losses#MAELoss)
- [MSELoss](https://github.com/TeaKatz/NLP_Losses/tree/main/src/nlp_losses#MSELoss)
- [HuberLoss](https://github.com/TeaKatz/NLP_Losses/tree/main/src/nlp_losses#HuberLoss)
- [CategoricalCrossEntropyLoss](https://github.com/TeaKatz/NLP_Losses/tree/main/src/nlp_losses#CategoricalCrossEntropyLoss)
- [BinaryCrossEntropyLoss](https://github.com/TeaKatz/NLP_Losses/tree/main/src/nlp_losses#BinaryCrossEntropyLoss)
- [CosineLoss](https://github.com/TeaKatz/NLP_Losses/tree/main/src/nlp_losses#CosineLoss)
- [LogisticLoss](https://github.com/TeaKatz/NLP_Losses/tree/main/src/nlp_losses#LogisticLoss)
- [NegativeLogisticLoss](https://github.com/TeaKatz/NLP_Losses/tree/main/src/nlp_losses#NegativeLogisticLoss)
- [TripletLoss](https://github.com/TeaKatz/NLP_Losses/tree/main/src/nlp_losses#TripletLoss)

## Requirements
- PyTorch (1.9.0 Recommended)

## Installation
    git clone https://github.com/TeaKatz/NLP_Losses
    cd NLP_Losses
    pip install --editable .

## Uninstallation
    pip uninstall nlp-losses

## Example
    import torch
    from nlp_losses import Losses

    loss_func = Losses(["TripletLoss"])

    anchor = torch.randn(100, 128, requires_grad=True)
    positive = torch.randn(100, 128, requires_grad=True)
    negative = torch.randn(100, 128, requires_grad=True)

    loss = loss_func(anchor, positive, negative)
    loss.backward()