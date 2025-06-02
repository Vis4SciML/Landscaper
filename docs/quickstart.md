# Using Landscaper

Before Landscaper can compute a loss landscape for your model, you will need to define two functions: 1.) the generator function for PyHessian that calculates per-sample gradients for your dataset, and one function that calculates the loss of your entire dataset. Leaving these functions as input parameters allows Landscaper to work with a wide range of models with minimal tinkering.

## PyHessian
We begin our loss landscape analysis by importing the `LossLandscape` and `PyHessian` classes and building a calculator object for the Hessian. 

```
from landscaper import LossLandscape, PyHessian

# data should be your input data and labels as tensors!
hessian_comp = PyHessian(model, model_criterion, data, device)
```

If we look at the definition for the `PyHessian` class, we can see that there's an additional keyword parameter called `hessian_generator`. We provide a generic implementation that should work for most torch models, but this can be adjusted to accomodate custom models. If the default implementation doesn't work for your model, try defining a generator function
```
def my_hessian_generator(model, criterion, data, device)
    ...
    yield input_size, model
```
where the gradient is saved in the model's parameters (the default uses `backward`) and each iteration yields the size of the input and a pointer to the model.  

## Defining a loss function
Once our hessian calculator is set up, we have to define a loss function for our model to be used when parameters are perturbed. Here's an example implementation:
```
def loss_function(model, data):
    total = 0.0
    for d in data:
        input, label = d
        output = model.forward(d)
        loss = criterion(output, label)
        total += loss
    return total
```
The function should take your model and data and return a total loss. With these elements in place, we can finally call `compute`:
```
landscape = LossLandscape.compute(model, data, hessian_comp, loss_function, dim=2, device=device)
```
and run various analysis and visualization methods!
