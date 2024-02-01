# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Quantize a model using intel extension for pytorch
"""

import argparse
import os
import logging
from pathlib import Path
import shutil

import intel_extension_for_pytorch as ipex
from neural_compressor.experimental import Quantization, common
from neural_compressor.model.torch_model import IPEXModel
from sklearn.metrics import accuracy_score
import torch
from transformers import BertTokenizerFast

from utils.dataloader import load_dataset
from utils.model import IntentAndTokenClassifier


class INCDataset:
    """Dataset wrapper for INC
    """

    def __init__(self, dloader, n_elements=None):
        self.dloader = dloader
        self.n_elements = n_elements

    def __getitem__(self, index):
        item = self.dloader[index]

        x_vals = {
            "input_ids": item['input_ids'],
            "attention_mask": item['attention_mask']
        }
        y_vals = (item['labels'], item['class_label'])

        return (x_vals), y_vals

    def __len__(self):
        if self.n_elements is None:
            return len(self.dloader)
        return self.n_elements


def main(flags) -> None:
    """Calibrate model for int 8 and serialize as a .pt

    Args:
        flags: benchmarking flags
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    if not os.path.exists(flags.saved_model):
        logger.error("Saved model %s not found!", flags.saved_model)
        return
    if not os.path.exists(flags.inc_config):
        logger.error("INC configuration %s not found!", flags.inc_config)
        return

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    try:
        dataset = load_dataset(flags.dataset_dir, tokenizer, flags.length)
    except FileNotFoundError as exc:
        logger.error("Please follow instructions to download data.")
        logger.error(exc, exc_info=True)
        return

    # set jit inputs for tracing
    jit_inputs = (
        dataset['test'][0]['input_ids'].view(1, -1),
        dataset['test'][0]['attention_mask'].view(1, -1)
    )

    def evaluate_accuracy(modelq: IntentAndTokenClassifier) -> float:
        """Evaluate accuracy of the provided pytorch model

        Args:
            modelq (IntentAndTokenClassifier): convai model

        Returns:
            float: combined NER + SQ class accuracy
        """
        tr_tk_preds, tr_tk_labels = [], []
        tr_sq_preds, tr_sq_labels = [], []

        for i in range(min(flags.quant_samples, len(dataset['test']))):
            ids = dataset['test'][i]['input_ids']
            mask = dataset['test'][i]['attention_mask']
            labels = dataset['test'][i]['labels']
            class_label = dataset['test'][i]['class_label']

            with torch.no_grad():
                out = modelq(
                    input_ids=ids.view(1, -1),
                    attention_mask=mask.view(1, -1)
                )

            tr_tk_logits = out[0].detach()
            tr_sq_logits = out[1].detach()

            # compute batch accuracy for token classification
            flattened_targets = labels.view(-1)
            active_logits = tr_tk_logits.view(-1, tr_tk_logits.shape[2])
            flattened_predictions = torch.argmax(active_logits, axis=1)

            # only get predictions of relevant tags
            active_accuracy = labels.view(-1) != -100
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(
                flattened_predictions,
                active_accuracy
            )

            tr_tk_labels.extend(labels.numpy())
            tr_tk_preds.extend(predictions.numpy())

            # compute accuracy for seqeunce classification
            predictions_sq = torch.argmax(tr_sq_logits, axis=1)
            if isinstance(class_label, list):
                tr_sq_labels.extend(class_label.numpy())
            else:
                tr_sq_labels.append(class_label.numpy())
            tr_sq_preds.extend(predictions_sq.numpy())

        tk_acc = accuracy_score(tr_tk_labels, tr_tk_preds)
        sq_acc = accuracy_score(tr_sq_labels, tr_sq_preds)
        res = (tk_acc + sq_acc)/2
        return res

    def eval_func(model_in: IntentAndTokenClassifier) -> float:
        """Evaluation function for use in INC

        Args:
            model_in (IntentAndTokenClassifier): base pytorch model. for ipex,
                it is unquantized and must be quantized using the
                config.

        Returns:
            float: accuracy of given model_in
        """
        use_int8 = False
        for path, _, files in os.walk('nc_workspace'):
            if 'ipex_config_tmp.json' in files:
                fullpath = os.path.join(path, 'ipex_config_tmp.json')
                use_int8 = True
                break

        if use_int8:
            model_in.eval()
            conf = ipex.quantization.QuantConf(configure_file=fullpath)
            modelq = ipex.quantization.convert(model_in, conf, jit_inputs)

        else:
            model_in.eval()
            modelq = model_in

        return evaluate_accuracy(modelq)

    # load the model
    model = IntentAndTokenClassifier(
        num_token_labels=len(dataset['train'].tag2id),
        num_sequence_labels=len(dataset['train'].class2id)
    )
    model.load_state_dict(torch.load(flags.saved_model))

    # remove previous runs of INC
    shutil.rmtree('nc_workspace', ignore_errors=True)

    # quantize model using provided configuration
    logger.info("Beginning calibration...")
    quantizer = Quantization(flags.inc_config)
    cmodel = common.Model(model)
    quantizer.model = cmodel
    quantizer.calib_dataloader = common.DataLoader(
        INCDataset(dataset['train'], 100), batch_size=20)
    quantizer.eval_func = eval_func
    quantized_model = quantizer.fit()

    # save the quantized model as a .pt
    if isinstance(quantized_model, IPEXModel):

        # ipex models save as configs, which need to be saved to a serialized model
        fname = Path(flags.saved_model).stem
        quantized_model.save(flags.output_dir)

        jit_inputs = (
            dataset['test'][0]['input_ids'].view(1, -1),
            dataset['test'][0]['attention_mask'].view(1, -1)
        )

        conf = ipex.quantization.QuantConf(
            flags.output_dir + "/best_configure.json")
        model = ipex.quantization.convert(model, conf, jit_inputs)
        torch.jit.save(model, flags.output_dir + "/" + fname + "_inc_quant.pt")
    else:
        quantized_model.save(flags.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s',
                        '--saved_model',
                        required=True,
                        help="saved pytorch (.pt) model to quantize.",
                        type=str
                        )

    parser.add_argument('-o',
                        '--output_dir',
                        required=True,
                        help="directory to save quantized model to.",
                        type=str
                        )

    parser.add_argument('-l',
                        '--length',
                        default=512,
                        help="sequence length to use. defaults to 512.",
                        type=int
                        )

    parser.add_argument('-q',
                        '--quant_samples',
                        default=100,
                        help="number of samples to use for quantization. defaults to 100.",
                        type=int
                        )

    parser.add_argument('-c',
                        '--inc_config',
                        help="INC conf yaml.",
                        required=True
                        )
    
    parser.add_argument('-d',
                        '--dataset_dir',
                        default=None,
                        type=str,
                        required=True,
                        help="directory to dataset"
                        )
    
    FLAGS = parser.parse_args()

    main(FLAGS)
