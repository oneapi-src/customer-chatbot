ARG BASE_NAME=continuumio/miniconda3
ARG IDP_VERSION=${IDP_VERSION}
ARG BASE_TAG=22.11.1
FROM ${BASE_NAME}:${BASE_TAG} as devkitminimal

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends --fix-missing \
    gcc \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    numactl

SHELL ["/bin/bash", "-c"] 

FROM devkitminimal as devkitintelfull39

ENV CONDA_VERSION=23.5.0

RUN conda install -n base conda==${CONDA_VERSION} && \
    conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda create -n jupyter_server -c conda-forge nb_conda_kernels jupyter notebook -y && \
    conda clean -afy

ENV LISTEN_IP=0.0.0.0
EXPOSE ${PORT}

WORKDIR /workspace
COPY env env
COPY docker/envinstall_all.sh envinstall_all.sh
RUN bash envinstall_all.sh

FROM devkitintelfull39 as jupyter_server

RUN groupadd -r user && useradd -r -g user -m user 
USER $USER

RUN conda init bash
