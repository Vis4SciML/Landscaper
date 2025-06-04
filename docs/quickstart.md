# Using Landscaper

Before Landscaper can compute a loss landscape for your model, you will need to define two functions: 
    
1. A generator function for PyHessian that calculates per-sample gradients for your dataset
    
2. A function that calculates the loss of your entire dataset. 

Leaving these functions as input parameters allows Landscaper to work with a wide range of models with minimal tinkering.

## PyHessian
We begin our loss landscape analysis by importing the `LossLandscape` and `PyHessian` classes and building a calculator object for the Hessian. 

```python
from torch import nn, Tensor
from landscaper import LossLandscape, PyHessian

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)
criterion = nn.CrossEntropyLoss()
data = Tensor([[0.1]*10, [0.2]*10])  # Example input data
device = 'cpu'  # or 'cuda' if you have a GPU

# Create a Hessian calculator
hessian_comp = PyHessian(model, criterion, data, device)
```

If we look at the definition for the `PyHessian` class, we can see that there's an additional keyword parameter called `hessian_generator`. We provide a generic implementation that should work for most `PyTorch` models, but this can be adjusted to accommodate custom models. If the default implementation doesn't work for your model, try defining a generator function

```python
def my_hessian_generator(
    model: nn.Module,
    criterion: nn.Module,
    data: Tensor,
    device: str
) -> Generator[Tuple[int, nn.Module]]:
    """
    An optional generator function that yields the size of the input and a pointer to the model.

    Args:
        model (nn.Module): The model for which to compute the Hessian.
        criterion (nn.Module): The loss function used to compute gradients.
        data (Tensor): The input data for the model.
        device (str): The device on which the model is located ('cpu' or 'cuda').
    
    Yields:
        Tuple[int, nn.Module]: A tuple containing the size of the input and a pointer to the model.
    """
    for sample in data:
        sample = sample.to(device)
        yield sample.size(0), model
```

where the gradient is saved in the model's parameters (the default uses `backward`) and each iteration yields the size of the input and a pointer to the model.

## Defining a Loss Function
Once our hessian calculator is set up, we have to define a loss function for our model to be used when parameters are perturbed. Here's an example implementation:

```python
def loss_function(model: nn.Module, data: Tensor) -> float:
    """
    A function that computes the total loss for the model given the data.
    
    Args:
        model (nn.Module): The model for which to compute the loss.
        data (Tensor): The input data for the model.
    
    Returns:
        float: The total loss for the model.
    """
    total = 0.0
    for d in data:
        sample, label = d
        output = model.forward(sample)
        loss = criterion(output, label)
        total += loss
    return total
```

## Computing the Loss Landscape

The function should take your model and data and return a total loss. With these elements in place, we can finally call `compute`:

```python
landscape = LossLandscape.compute(
    model, 
    data, 
    hessian_comp,
    loss_function,
    dim=2,
    device=device
)
```
This will compute the loss landscape for your model using the provided data and hessian calculator. The `dim` parameter specifies the dimensionality of the perturbation space (2 for 2D landscapes, 3 for 3D landscapes, etc.).

## Visualizing the Landscape