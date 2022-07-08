# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Train the intent and classfier model
"""

import logging
from typing import Union

from sklearn.metrics import accuracy_score
import torch

from .model import IntentAndTokenClassifier

logger = logging.getLogger()


def evaluate_accuracy(
    dataloader: torch.utils.data.DataLoader,
    model: IntentAndTokenClassifier,
) -> Union[float, float]:
    """Evaluate the accuracy on the provided dataset

    Args:
        dataloader (torch.utils.data.DataLoader): dataloader to evaluate on
        model (IntentAndTokenClassifier): model to evaluate

    Returns:
        Union[float, float]: token prediction accuracy, class prediction
            accuracy
    """

    tr_tk_preds, tr_tk_labels = [], []
    tr_sq_preds, tr_sq_labels = [], []
    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(dataloader):

            ids = batch['input_ids']
            mask = batch['attention_mask']
            labels = batch['labels']
            class_label = batch['class_label']

            # pass inputs through model
            out = model(
                input_ids=ids,
                attention_mask=mask,
                token_labels=labels,
                sequence_labels=class_label)

            tr_tk_logits = out[0]
            tr_sq_logits = out[2]

            # compute batch accuracy for token classification
            flattened_targets = labels.view(-1)
            active_logits = tr_tk_logits.view(-1, model.num_token_labels)
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
            tr_sq_labels.extend(class_label.numpy())
            tr_sq_preds.extend(predictions_sq.numpy())

    return (
        accuracy_score(tr_tk_labels, tr_tk_preds),
        accuracy_score(tr_sq_labels, tr_sq_preds)
    )


def train(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,
        max_grad_norm: float = 10) -> None:
    """train a model on the given dataset

    Args:
        dataloader (torch.utils.data.DataLoader): training dataset
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer to use
        epochs (int, optional): number of training epochs. Defaults to 5.
        max_grad_norm (float, optional): gradient clipping. Defaults to 10.
    """

    model.train()

    for epoch in range(1, epochs + 1):

        running_loss = 0
        tr_tk_preds, tr_tk_labels = [], []
        tr_sq_preds, tr_sq_labels = [], []

        for idx, batch in enumerate(dataloader):

            optimizer.zero_grad()

            ids = batch['input_ids']
            mask = batch['attention_mask']
            labels = batch['labels']
            class_label = batch['class_label']

            # pass inputs through model
            out = model(
                input_ids=ids,
                attention_mask=mask,
                token_labels=labels,
                sequence_labels=class_label
            )

            # evaluate loss
            token_loss = out[1]
            sequence_loss = out[3]
            combined_loss = token_loss + sequence_loss

            running_loss += combined_loss.item()

            tr_tk_logits = out[0]
            tr_sq_logits = out[2]

            if idx % 100 == 0:
                logger.info("loss/100 batches: %.4f", running_loss/(idx + 1))

            # compute batch accuracy for token classification
            flattened_targets = labels.view(-1)
            active_logits = tr_tk_logits.view(-1, model.num_token_labels)
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
            tr_sq_labels.extend(class_label.numpy())
            tr_sq_preds.extend(predictions_sq.numpy())

            # clip gradients for stability
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
            )

            combined_loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(dataloader)
        logger.info(
            "Training loss epoch #%d : %.4f", epoch, epoch_loss
        )
        logger.info(
            "Training NER accuracy epoch #%d : %.4f",
            epoch,
            accuracy_score(tr_tk_labels, tr_tk_preds)
        )
        logger.info(
            "Training CLS accuracy epoch #%d : %.4f",
            epoch,
            accuracy_score(tr_sq_labels, tr_sq_preds)
        )
