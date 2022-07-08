# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run inference benchmarks using openvino optimized model.
"""

import argparse
import logging
import os
import tempfile
import time

import numpy as np
import openvino.runtime as ov
import torch
from transformers import BertTokenizerFast

from utils.dataloader import load_dataset
from utils.model import IntentAndTokenClassifier


def inference(predict_fn, ids, mask, n_runs) -> float:
    """Run inference using the provided `predict_fn`

    Args:
        predict_fn: prediction function to use
        ids: input ids to feed into the predict function
        mask: attention masks to feed into the predict function
        n_runs: number of benchmark runs to time
    
    Returns:
        float : Average prediction time
    """
    times = []
    for _ in range(2+n_runs):
        with torch.no_grad():
            start = time.time()
            predict_fn(ids, mask)
            end = time.time()
        times.append(end - start)

    avg_time = np.mean(times[2:])
    return avg_time


def main(flags) -> None:
    """Setup model for inference and perform benchmarking

    Args:
        flags: benchmarking flags
    """
    if flags.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(filename=flags.logfile, level=logging.DEBUG)
    logger = logging.getLogger()

    if not os.path.exists(flags.saved_model):
        logger.error("Saved model %s not found!", flags.saved_model)
        return

    # batch size and seq length for inference are not fixed model
    # parameters for BERT models
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    try:
        dataset = load_dataset("../data/atis-2/", tokenizer, flags.length)
    except FileNotFoundError as exc:
        logger.error("Please follow instructions to download data.")
        logger.error(exc, exc_info=True)
        return

    items = [dataset['test'][i] for i in range(flags.batch_size)]
    ids = torch.stack([item['input_ids'] for item in items])
    mask = torch.stack([item['attention_mask'] for item in items])
    ids = ids.view(flags.batch_size, -1)
    mask = mask.view(flags.batch_size, -1)

    if flags.is_pytorch_model:

        logger.info("Converting model %s", flags.saved_model)
        # load saved, model, data and tokenizer
        model = IntentAndTokenClassifier(
            num_token_labels=len(dataset['train'].tag2id),
            num_sequence_labels=len(dataset['train'].class2id)
        )
        model.load_state_dict(torch.load(flags.saved_model))
        model.eval()

        # create an onnx model
        with tempfile.NamedTemporaryFile() as tmp:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (ids[0:1], mask[0:1]),
                    tmp.name,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=["logits_token", "sequence_loss"],
                    dynamic_axes={
                        'input_ids': {0: "batch_size"},
                        'attention_mask': {0: "batch_size"}
                    }
                )

        # read in onnx and convert to openVINO IR
        core = ov.Core()
        model = core.read_model(tmp.name)
    else:

        logger.info("Loading model %s", flags.saved_model)
        core = ov.Core()
        model = core.read_model(flags.saved_model)

    inlayer_iter = iter(model.inputs)
    input_layer1 = next(inlayer_iter)
    input_layer2 = next(inlayer_iter)
    model.reshape({
        input_layer1.any_name: ids.shape,
        input_layer2.any_name: mask.shape
    })
    compiled_model = core.compile_model(model, 'CPU')

    def predict(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Predicts the output for the given `input_ids` and `attention_mask`
        using the `exec_net_ir` OpenVINO model.

        Args:
            input_ids (torch.Tensor): input_ids Tensor from transformers
            tokenizer attention_mask (torch.Tensor): attention mask Tensor
            from transformers tokenizer

        Returns:
            torch.Tensor: predicted quantities
        """

        return compiled_model(inputs=[input_ids, attention_mask])

    logger.info("Running experiment n = %d, b = %d, l = %d",
                flags.n_runs, flags.batch_size, flags.length)
    average_time = inference(predict, ids, mask, flags.n_runs)
    logger.info('Avg time per batch : %.3f s', average_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-s',
                        '--saved_model',
                        required=True,
                        help="OV(.xml) or pytorch(.pt) model to benchmark",
                        type=str
                        )

    parser.add_argument('--is_pytorch_model',
                        default=False,
                        action="store_true",
                        help="flag if saved model is .pt. defaults to true.",
                        )

    parser.add_argument('-b',
                        '--batch_size',
                        default=200,
                        help="batch size to use. defaults to 200.",
                        type=int
                        )

    parser.add_argument('-l',
                        '--length',
                        default=512,
                        help="sequence length to use. defaults to 512.",
                        type=int
                        )
   
    parser.add_argument('--logfile',
                        help="logfile to use.",
                        default="",
                        type=str
                        )

    parser.add_argument('-n',
                        '--n_runs',
                        default=100,
                        help="number of trials to test. defaults to 100.",
                        type=int
                        )

    FLAGS = parser.parse_args()

    main(FLAGS)
