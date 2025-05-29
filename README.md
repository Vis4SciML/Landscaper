<div align="center">

<img src="assets/logo.png" width="600">
<br>
<h3>A comprehensive Python framework designed for exploring the loss landscapes of deep learning models.</h3> 

</div>

## Introduction

`Landscaper` is a comprehensive Python framework designed for exploring the loss landscapes of deep learning models. It integrates three key functionalities:

- **Construction**: Builds detailed loss landscape representations through low and high-dimensional sampling.
- **Quantification**: Applies advanced metrics, including a novel topological data analysis (TDA) based smoothness metric, enabling new insights into model behavior.
- **Visualization**: Offers intuitive tools to visualize and interpret loss landscapes, providing actionable insights beyond traditional performance metrics.

By uniting these aspects, Landscaper empowers users to gain a deeper, more holistic understanding of their model's behavior. More information can be found in the [documentation] or in the [paper].

## Quick Start

Check out the [quick start guide] for a step-by-step introduction to using Landscaper.

## Documentation
The full documentation for Landscaper is available at [placeholder]. It includes detailed instructions on installation, usage, and examples.

## Installation
Landscaper is available on [PyPI](https://pypi.org/project/landscaper/), making it easy to install and integrate into your projects.
s
Landscaper requires Python `>=3.10,<3.13` and PyTorch `>=2.0.0` and is compatible with both CPU and GPU environments. To install PyTorch, follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to select the appropriate version for your system. Then you can install Landscaper using pip. 

To install Landscaper, run the following command:

```bash
pip install landscaper
```

To install Landscaper with all the dependencies to run the examples, use:

```bash
pip install landscaper[examples]
```

## BibTeX Citation 
If you use Landscaper in your research, please consider citing it. You can use the following BibTeX entry:

```
@article{CITE_KEY,
  title = {},
  author = {},
  year = {},
  month = {},
  journal = {},
  volume = {},
  number = {},
  pages = {},
  publisher = {},
  issn = {},
  doi = {},
}
```

## Developers
Install the dev dependencies with `uv sync`. When running `pytest`, pass `--html=report.html` to be able to visualize images created by the tests.
