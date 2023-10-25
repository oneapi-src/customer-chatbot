#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

echo 'Search for requirements.yml files'
env_paths=$(find -wholename './env/*.yml')
echo Found $env_paths
for i in $env_paths
do 
    echo Will install $i
    env_name=$(cat $i | grep -Po "(?<=name:)(.*)")
    conda env create -f $i 
    conda install -n $env_name nb_conda_kernels
done

