<div align="center">

<img src="assets/logo.png" width="600">
<br>
<h3>A comprehensive Python framework designed for exploring the loss landscapes of deep learning models.</h3> 
<a href="https://doi.org/10.5281/zenodo.15874987"><img src="https://zenodo.org/badge/978321484.svg" alt="DOI"></a>
</div>


## Introduction

`Landscaper` is a comprehensive Python framework designed for exploring the loss landscapes of deep learning models. It integrates three key functionalities:

- **Construction**: Builds detailed loss landscape representations through low and high-dimensional sampling, including Hessian analysis adapted from [PyHessian](https://github.com/amirgholami/PyHessian).
- **Quantification**: Applies advanced metrics, including a novel topological data analysis (TDA) based smoothness metric, enabling new insights into model behavior.
- **Visualization**: Offers intuitive tools to visualize and interpret loss landscapes, providing actionable insights beyond traditional performance metrics.

By uniting these aspects, Landscaper empowers users to gain a deeper, more holistic understanding of their model's behavior. More information can be found in the [documentation] or in the [paper].

## Quick Start

Check out the [quick start guide](https://vis4sciml.github.io/Landscaper/quickstart/) for a step-by-step introduction to using Landscaper.

## Documentation
The full documentation for Landscaper is available at [https://vis4sciml.github.io/Landscaper/](https://vis4sciml.github.io/Landscaper/#). It includes detailed instructions on installation, usage, and examples. You can build the docs locally by installing the `docs` extras and then running `mkdocs serve`.

## Installation
Landscaper is available on [PyPI](https://pypi.org/project/landscaper/), making it easy to install and integrate into your projects.

Landscaper requires Python `>=3.10,<3.13` and PyTorch `>=2.0.0` and is compatible with both CPU and GPU environments. To install PyTorch, follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to select the appropriate version for your system. Then you can install Landscaper using pip. 

To install Landscaper, run the following command:

```bash
pip install landscaper
```

To install Landscaper with all the dependencies to run the examples, use:

```bash
pip install landscaper[examples]
```

### Docker

Additionally, Landscaper provides a Docker image that includes all dependencies, based on the official [PyTorch image](https://hub.docker.com/layers/pytorch/pytorch/2.6.0-cuda12.4-cudnn9-runtime/images/sha256-77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee). You can pull the image from the GitHub Container Registry:

```bash
docker pull ghcr.io/vis4sciml/landscaper:latest
```

## BibTeX Citation 
If you use Landscaper in your research, please consider citing it. You can use the following BibTeX entry:

```
@misc{https://doi.org/10.5281/zenodo.15874987,
  doi = {10.5281/ZENODO.15874987},
  url = {https://zenodo.org/doi/10.5281/zenodo.15874987},
  author = {Jiaqing Chen and Nicholas Hadler and Tiankai Xie and Rostyslav Hnatyshyn},
  title = {Vis4SciML/Landscaper},
  publisher = {Zenodo},
  year = {2025},
  copyright = {Lawrence Berkeley National Labs BSD Variant License}
}
```

## Developers
Install the dev dependencies with `uv sync`. When running `pytest`, pass `--html=report.html` to be able to visualize images created by the tests.

## Copyright

Landscaper Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy), University of California, Berkeley,  and Arizona State University.  All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
