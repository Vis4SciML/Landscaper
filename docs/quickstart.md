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
    A generator function that yields the size of each input sample and its gradient.

    Args:
        model (nn.Module): The model for which to compute the Hessian.
        criterion (nn.Module): The loss function, used to compute gradients.
        data (Tensor): The input data for the model.
        device (str): The device on which the model is located ('cpu' or 'cuda').
    
    Yields:
        Tuple[int, nn.Module]: A tuple containing the size of the input and the gradient of each sample.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    for sample, target in data: 
        outputs = model.forward(sample)
        loss = criterion(outputs, targets) 
        
        grads = torch.autograd.grad(
            loss, params, create_graph=True, materialize_grads=True
        )
        yield sample.size(0), grads
```

where each iteration yields the size of the input and the gradient of the loss. Most of the time, you will only need to change how the loss is being calculated or how the data is being accessed.

## Defining a Scalar Function
Once our hessian calculator is set up, we have to define a function that takes a model and our data. This function gets called for every coordinate in the loss landscape with a perturbed version of our model. Here's an example use that calculates the average loss for a model:

```python
def scalar_function(model: nn.Module, data: Tensor) -> float:
    """
    A function that computes the average loss for the model given the data.
    
    Args:
        model (nn.Module): The model to compute the loss with.
        data (Tensor): The input data for the model.
    
    Returns:
        float: The average loss for the model.
    """
    total = 0.0
    count = 0
    for d in data:
        sample, label = d
        output = model.forward(sample)
        loss = criterion(output, label)
        total += loss
        count += 1
    return total / count
```

## Computing the Loss Landscape

With these elements in place, we can finally call `compute`:

```python

directions = hessian_comp.eigenvalues(top_n=3)

landscape = LossLandscape.compute(
    model, 
    data,
    directions,
    hessian_comp,
    loss_function,
    dim=2,
    device=device
)
```
This will compute the loss landscape for your model using the provided data and hessian calculator. The `dim` parameter specifies the dimensionality of the perturbation space (2 for 2D landscapes, 3 for 3D landscapes, etc.).

## Visualizing the Landscape

The landscape can be visualized in a number of different ways once it is finally computed.

```python

landscape.show() # shows a 3D render of the landscape if dim=2
landscape.show_profile() # shows a 1D landscape profile
landscape.show_contour() # contour plot
landscape.show_persistence_barcode() # persistence barcode
```

If you are interested in examining the merge tree, you can visualize it using networkx:
```python
from Landscaper.tda import digraph_mt
import networkx as nx

mt = landscape.get_sublevel_tree() # gets the minima merge tree
# mt = landscape.get_super_tree()

g = digraph_mt(mt)
nx.draw_planar(g) # draws a planar graph of the merge tree
```
