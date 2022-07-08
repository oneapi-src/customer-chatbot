#!/usr/bin/bash

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

torch-model-archiver --model-name convai --version 1.0 --serialized-file ../saved_models/intel/convai_int8.pt --handler custom_handler.py

cd serve/benchmark
python benchmark-ab.py --config config.json
