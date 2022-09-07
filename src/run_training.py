# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run training benchmarks
"""

import argparse
import logging
import os
import pathlib
import time

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from utils.dataloader import load_dataset
from utils.model import IntentAndTokenClassifier
from utils.train import evaluate_accuracy, train

# training parameters
MAX_LENGTH = 64
BATCH_SIZE = 20
EPOCHS = 3
MAX_GRAD_NORM = 10

torch.manual_seed(0)


def main(flags):
    """Benchmark model training

    Args:
        flags: benchmarking configuration
    """

    if flags.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(flags.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=path, level=logging.DEBUG)
    logger = logging.getLogger()

    # Create tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Read in the datasets and crate dataloaders
    logger.debug("Reading in the data...")

    try:
        dataset = load_dataset("../data/atis-2/", tokenizer, MAX_LENGTH)
    except FileNotFoundError as exc:
        logger.error("Please follow instructions to download data.")
        logger.error(exc, exc_info=True)
        return

    train_loader = DataLoader(
        dataset['train'], batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE)

    # Create model and prepare for training
    start = time.time()
    model = IntentAndTokenClassifier(
        num_token_labels=len(dataset['train'].tag2id),
        num_sequence_labels=len(dataset['train'].class2id)
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)

    model.train()

    # if using intel, optimize the model and the optimizer
    if flags.intel:
        import intel_extension_for_pytorch as ipex
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    # Train the model
    logger.debug("Training the model...")
    train(train_loader, model, optimizer, epochs=EPOCHS,
          max_grad_norm=MAX_GRAD_NORM)
    training_time = time.time()

    # Evaluate accuracy on the testing set in batches
    accuracy_ner, accuracy_class = evaluate_accuracy(test_loader, model)
    testing_time = time.time()

    # Save model
    if flags.save_model_dir:
        path = pathlib.Path(flags.save_model_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path / "convai.pt")

        if flags.save_onnx:
            # prepare data for onnx conversion
            items = [dataset['test'][i] for i in range(flags.n_stat)]
            ids = torch.stack([item['input_ids'] for item in items])
            ids = ids.view(flags.n_stat, -1)
            mask = torch.stack([item['attention_mask'] for item in items])
            mask = mask.view(flags.n_stat, -1)

            # create an onnx model
            onnx_model = os.path.join(flags.save_model_dir, "model.onnx")
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (ids[0:1], mask[0:1]),
                    onnx_model,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=["logits_token", "logits_sequence"],
                    dynamic_axes={
                        'input_ids': {0: "batch_size"},
                        'attention_mask': {0: "batch_size"}
                    }
                )

    logger.info("=======> Test Accuracy on NER : %.2f", accuracy_ner)
    logger.info("=======> Test Accuracy on CLS : %.2f", accuracy_class)
    logger.info("=======> Training Time : %.3f secs", training_time - start)
    logger.info("=======> Inference Time : %.3f secs",
                testing_time - training_time)
    logger.info("=======> Total Time: %.3f secs", testing_time - start)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")

    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies where available"
                        )

    parser.add_argument('-s',
                        '--save_model_dir',
                        default=None,
                        type=str,
                        required=False,
                        help="directory to save model under"
                        )

    parser.add_argument('--save_onnx',
                        default=False,
                        action="store_true",
                        required=False,
                        help="also export an ONNX model"
                        )

    FLAGS = parser.parse_args()

    main(FLAGS)
