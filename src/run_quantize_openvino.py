# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run inference benchmarks using openvino optimized model.
"""

import argparse
import os

import numpy as np
from openvino.tools.pot.api import Metric, DataLoader
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.pipeline.initializer import create_pipeline
from sklearn.metrics import accuracy_score
import torch
from transformers import BertTokenizerFast

from utils.dataloader import load_dataset

class LPOTDataLoader(DataLoader):
    """Data loader for LPOT.
    """

    def __init__(self, ids, masks, labels, class_labels):
        self.ids = ids
        self.masks = masks
        self.labels = labels
        self.class_labels = class_labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return (
            (index, (self.labels[index], self.class_labels[index])),
            {"input_ids": self.ids[index], "attention_mask": self.masks[index]}
        )


class CombinedAccuracy(Metric):
    """Accuracy metric for LPOT
    """

    def __init__(self):
        super().__init__()
        self._name = "combined_accuracy"
        self._tr_tk_preds, self._tr_tk_labels = [], []
        self._tr_sq_preds, self._tr_sq_labels = [], []
        self._combined_acc = []

    @property
    def value(self):
        tk_acc = accuracy_score(self._tr_tk_labels, self._tr_tk_preds)
        sq_acc = accuracy_score(self._tr_sq_labels, self._tr_sq_preds)
        return {self._name: (tk_acc + sq_acc)/2}

    @property
    def avg_value(self):
        return {self._name: np.mean(self._combined_acc)}

    def update(self, output, target):
        tr_tk_logits = torch.as_tensor(output[0])
        tr_sq_logits = torch.as_tensor(output[1])
        labels = target[0][0]
        class_label = target[0][1]

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
        self._tr_tk_labels.extend(labels.numpy())
        self._tr_tk_preds.extend(predictions.numpy())

        # compute accuracy for seqeunce classification
        predictions_sq = torch.argmax(tr_sq_logits, axis=1)
        if isinstance(class_label, list):
            self._tr_sq_labels.extend(class_label.numpy())
        else:
            self._tr_sq_labels.append(class_label.numpy())
        self._tr_sq_preds.extend(predictions_sq.numpy())

        tk_acc = accuracy_score(labels, predictions)
        sq_acc = class_label == predictions_sq
        self._combined_acc.append((tk_acc + sq_acc)/2)

    def get_attributes(self):
        return {self._name:
                {"direction": "higher-better", "type": "accuracy"}
                }

    def reset(self):
        self._tr_tk_preds, self._tr_tk_labels = [], []
        self._tr_sq_preds, self._tr_sq_labels = [], []
        self._combined_acc = []


def main(flags) -> None:
    """Setup model for inference and perform benchmarking

    Args:
        flags: benchmarking flags
    """

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    try:
        dataset = load_dataset("../data/atis-2/", tokenizer, flags.length)
    except FileNotFoundError as exc:
        print("Please follow instructions to download data.")
        print(exc)
        return

    # prepare data
    items = [dataset['test'][i] for i in range(flags.n_stat)]
    ids = torch.stack([item['input_ids'] for item in items])
    ids = ids.view(flags.n_stat, -1)
    mask = torch.stack([item['attention_mask'] for item in items])
    mask = mask.view(flags.n_stat, -1)
    labels = torch.stack([item['labels'] for item in items])
    class_labels = torch.stack([item['class_label'] for item in items])

    # create a temporary OV IR model
    #_ = subprocess.run([
    #    "mo",
    #    "--input_model", onnx_model,
    #    "--input_shape", f"[{1},{ids.shape[1]}],[{1},{mask.shape[1]}]",
    #    "--input", "input_ids,attention_mask",
    #    "--data_type", "FP32",
    #    "--output_dir", tmpdir
    #], check=True)

    # configure quantization algorithm
    model_config = {
        "model_name": "convai-model",
        "model": os.path.join(flags.ir_model_dir, "model.xml"),
        "weights": os.path.join(flags.ir_model_dir, "model.bin")
    }

    engine_config = {
        "device": "CPU",
    }

    algorithms = [
        {
            "name": "DefaultQuantization",
            "params": {
                    "target_device": "CPU",
                    "model_type": "transformer",
                    "preset": "performance",
                    "stat_subset_size": flags.n_stat
            }
        }
    ]

    # evaluate and quantize model
    ir_model = load_model(model_config=model_config)

    data_loader = LPOTDataLoader(ids, mask, labels, class_labels)
    metric = CombinedAccuracy()
    engine = IEEngine(
        config=engine_config,
        data_loader=data_loader,
        metric=metric
    )
    pipeline = create_pipeline(algo_config=algorithms, engine=engine)
    fp_results = pipeline.evaluate(ir_model)
    if fp_results:
        print("FP32 model results:")
        for name, value in fp_results.items():
            print(f"{name} : {value:.5f}")

    compressed_model = pipeline.run(ir_model)
    compress_model_weights(model=compressed_model)

    save_model(
        model=compressed_model,
        save_path=flags.output_dir,
        model_name="convai_ov_int8")

    int_results = pipeline.evaluate(compressed_model)
    if int_results:
        print("INT8 model results:")
        for name, value in int_results.items():
            print(f"{name} : {value:.5f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--ir_model_dir',
                        required=True,
                        help="directory of OpenVINO IR model.",
                        type=str
                        )

    parser.add_argument('-l',
                        '--length',
                        default=512,
                        help="sequence length to use. defaults to 512.",
                        type=int
                        )

    parser.add_argument('-o',
                        '--output_dir',
                        required=True,
                        help="directory to save quantized model",
                        type=str
                        )

    parser.add_argument('-n',
                        '--n_stat',
                        default=100,
                        help="# of samples to use for quant. defaults to 100.",
                        type=int
                        )

    FLAGS = parser.parse_args()

    if not os.path.exists(FLAGS.saved_model):
        print("Saved model not found!")
    else:
        main(FLAGS)
