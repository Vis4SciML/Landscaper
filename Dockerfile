# syntax=docker/dockerfile:1

LABEL maintainer="Jiaqing Chen <jchen501@asu.edu>, Nicholas Hadler <nhadler@berkeley.edu>, Tiankai Xie <txie21@asu.edu>, Rostyslav Hnatyshyn <rhnatysh@asu.edu>"
LABEL org.opencontainers.image.description="Landscaper docker image with PyTorch 2.6.0 with CUDA 12.4."
LABEL org.opencontainers.image.source=https://github.com/Vis4SciML/Landscaper
LABEL org.opencontainers.image.licenses=GPL-3.0

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee
COPY --from=ghcr.io/astral-sh/uv:0.6.16@sha256:db305ce8edc1c2df4988b9d23471465d90d599cc55571e6501421c173a33bb0b  /uv /uvx /bin/

# Install build essentials for compiling C/C++ dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set the environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_PROJECT_ENVIRONMENT=/opt/conda/ 

WORKDIR /app

COPY . .

RUN uv pip install . 