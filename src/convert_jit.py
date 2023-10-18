# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run inference benchmarks
"""

import argparse
import os

import torch
from transformers import BertTokenizerFast

from utils.dataloader import load_dataset
from utils.model import IntentAndTokenClassifier


def main(flags) -> None:
    """Setup model for inference and perform benchmarking

    Args:
        FLAGS: benchmarking flags
    """

    # batch size and seq length for inference are not fixed model
    # parameters for BERT models
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = load_dataset(flags.dataset_dir, tokenizer, 512)
    items = [dataset['test'][i] for i in range(1)]
    ids = torch.stack([item['input_ids'] for item in items])
    mask = torch.stack([item['attention_mask'] for item in items])

    if flags.is_inc_int8:
        from neural_compressor.utils.pytorch import load

        model = IntentAndTokenClassifier(
            num_token_labels=len(dataset['train'].tag2id),
            num_sequence_labels=len(dataset['train'].class2id)
        )

        model = load(flags.saved_model_dir, model)
    else:
        model = IntentAndTokenClassifier(
            num_token_labels=len(dataset['train'].tag2id),
            num_sequence_labels=len(dataset['train'].class2id)
        )
        model.load_state_dict(
            torch.load(os.path.join(flags.saved_model_dir, "convai.pt"))
        )

    ids = ids.view(1, -1)
    mask = mask.view(1, -1)

    model.eval()
    model = torch.jit.trace(
        model,
        [
            dataset['test'][0]['input_ids'].view(1, -1),
            dataset['test'][0]['attention_mask'].view(1, -1)
        ],
        check_trace=False,
        strict=False
    )
    model = torch.jit.freeze(model)
    torch.jit.save(model, flags.output_model)
    print(f"Model exported to: {flags.output_model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s',
                        '--saved_model_dir',
                        required=True,
                        help="directory of saved model to benchmark.",
                        type=str
                        )

    parser.add_argument('-o',
                        '--output_model',
                        required=True,
                        help="saved torchscript (.pt) model",
                        type=str
                        )
    
    parser.add_argument('-d',
                        '--dataset_dir',
                        default=None,
                        type=str,
                        required=True,
                        help="directory to dataset"
                        )
    
    parser.add_argument(
        '--is_inc_int8',
        default=False,
        action="store_true",
        help="saved model dir is a quantized int8 model. defaults to False."
    )

    FLAGS = parser.parse_args()

    main(FLAGS)
