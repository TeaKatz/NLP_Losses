# Losses

## MAELoss:
> **PARAMETERS:**
>
> **inputs** (*Tensor*) - Prediction output
>
> **targets** (*Tensor*) - Target prediction
>
> **reduction** (*str*, *optional*) - Reduction method apply to the output: `"sum"` | `"mean"` | `"none"`. Default: `"sum"`

## MSELoss:
> **PARAMETERS:**
>
> **inputs** (*Tensor*) - Prediction output
>
> **targets** (*Tensor*) - Target prediction
>
> **reduction** (*str*, *optional*) - Reduction method apply to the output: `"sum"` | `"mean"` | `"none"`. Default: `"sum"`

## HuberLoss:
> **PARAMETERS:**
>
> **inputs** (*Tensor*) - Prediction output
>
> **targets** (*Tensor*) - Target prediction
>
> **reduction** (*str*, *optional*) - Reduction method apply to the output: `"sum"` | `"mean"` | `"none"`. Default: `"sum"`

## CategoricalCrossEntropyLoss:
> **PARAMETERS:**
>
> **inputs** (*Tensor*) - Prediction output
>
> **targets** (*Tensor*) - Target prediction
>
> **reduction** (*str*, *optional*) - Reduction method apply to the output: `"sum"` | `"mean"` | `"none"`. Default: `"sum"`
>
> **ignore_index** (*int*, *optional*) - Index to ignore when compute gradient. Default: `-100`

## BinaryCrossEntropyLoss:
> **PARAMETERS:**
>
> **inputs** (*Tensor*) - Prediction output
>
> **targets** (*Tensor*) - Target prediction
>
> **reduction** (*str*, *optional*) - Reduction method apply to the output: `"sum"` | `"mean"` | `"none"`. Default: `"sum"`

## CosineLoss:
> **PARAMETERS:**
>
> **inputs** (*Tensor*) - Prediction output
>
> **targets** (*Tensor*) - Target prediction
>
> **margin** (*float*, *optional*) - Loss margin. Default: `0.0`
>
> **reduction** (*str*, *optional*) - Reduction method apply to the output: `"sum"` | `"mean"` | `"none"`. Default: `"sum"`

## LogisticLoss:
> **PARAMETERS:**
>
> **inputs** (*Tensor*) - Prediction output
>
> **targets** (*Tensor*) - Target prediction
>
> **reduction** (*str*, *optional*) - Reduction method apply to the output: `"sum"` | `"mean"` | `"none"`. Default: `"sum"`

## NegativeLogisticLoss:
> **PARAMETERS:**
>
> **inputs** (*Tensor*) - Prediction output
>
> **targets** (*Tensor*) - Target prediction
>
> **reduction** (*str*, *optional*) - Reduction method apply to the output: `"sum"` | `"mean"` | `"none"`. Default: `"sum"`

## TripletLoss:
> **PARAMETERS:**
>
> **anchors** (*Tensor*) - Prediction output
>
> **positives** (*Tensor*) - Positive samples
>
> **negatives** (*Tensor*) - Negative samples
>
> **distance_function** (*callable*, *optional*) - Distance function, Default: `None`
>
> **margin** (*float*, *optional*) - Loss margin. Default: `0.0`
>
> **reduction** (*str*, *optional*) - Reduction method apply to the output: `"sum"` | `"mean"` | `"none"`. Default: `"sum"`