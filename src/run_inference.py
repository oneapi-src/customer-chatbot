# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run inference benchmarks
"""

import argparse
import logging
import os
import time
from tqdm import tqdm

import numpy as np
import torch
import intel_extension_for_pytorch as ipex
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
    predictions = []
    for _ in tqdm(range(2 + n_runs), desc="Running Inference: "):
        with torch.no_grad():
            start = time.time()
            res = predict_fn(ids, mask)
            end = time.time()
            predictions.append(res)
        times.append(end - start)

    avg_time = np.mean(times[2:])
    return avg_time


def main(flags) -> None:
    """Setup model for inference and perform benchmarking

    Args:
        FLAGS: benchmarking flags
    """

    if flags.logfile == "":
        logging.root.handlers = []
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.root.handlers = []
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[
                                logging.FileHandler(flags.logfile),
                                logging.StreamHandler()
                            ])
    logger = logging.getLogger()

    batch_size = flags.batch_size
    seq_length = flags.length

    if not os.path.exists(flags.saved_model_dir):
        logger.error("Saved model %s not found!", flags.saved_model_dir)
        return

    # batch size and seq length for inference are not fixed model
    # parameters for BERT models
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    try:
        dataset = load_dataset(flags.dataset_dir, tokenizer, seq_length)
    except FileNotFoundError as exc:
        logger.error("Please follow instructions to download data.")
        logger.error(exc, exc_info=True)
        return

    items = [dataset['test'][i] for i in range(batch_size)]
    ids = torch.stack([item['input_ids'] for item in items])
    mask = torch.stack([item['attention_mask'] for item in items])

    # load saved, model, data and tokenizer
    if flags.is_jit:
        model = torch.jit.load(
            os.path.join(flags.saved_model_dir, "convai.pt")
        )
    elif flags.is_inc_int8:
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

    if batch_size == 1:
        ids = ids.view(1, -1)
        mask = mask.view(1, -1)

    logger.info("Using IPEX to optimize model")

    model.eval()
    model = ipex.optimize(model)
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

    #cpu_pool = intel_extension_for_pytorch.cpu.runtime.CPUPool(node_id=0)
    # @intel_extension_for_pytorch.cpu.runtime.pin(cpu_pool)
    def predict(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Predicts the output for the given `input_ids` and
            `attention_mask` using the given PyTorch model.

        Args:
            input_ids (torch.Tensor): input_ids Tensor from transformers
                tokenizer
            attention_mask (torch.Tensor): attention mask Tensor from
                transformers tokenizer

        Returns:
            torch.Tensor: predicted quantities
        """

        return model(input_ids=input_ids, attention_mask=attention_mask)

    logger.info("Running experiment n = %d, b = %d, l = %d",
                flags.n_runs, flags.batch_size, flags.length)

    average_time = inference(predict, ids, mask, FLAGS.n_runs)
    logger.info('Avg time per batch : %.3f s', average_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s',
                        '--saved_model_dir',
                        required=True,
                        help="directory of saved model to benchmark.",
                        type=str
                        )

    parser.add_argument('--is_jit',
                        default=False,
                        action="store_true",
                        help="if the model is torchscript. defaults to False.",
                        )

    parser.add_argument(
        '--is_inc_int8',
        default=False,
        action="store_true",
        help="saved model dir is a quantized int8 model. defaults to False."
    )

    parser.add_argument('-b',
                        '--batch_size',
                        default=200,
                        help="batch size to use. defaults to 200.",
                        type=int
                        )
    
    parser.add_argument('-d',
                        '--dataset_dir',
                        default=None,
                        type=str,
                        required=True,
                        help="directory to dataset"
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
