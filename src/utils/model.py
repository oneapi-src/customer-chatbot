# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Intent and Token Classification Model built using BERT
"""

import torch
from transformers import BertModel

class IntentAndTokenClassifier(torch.nn.Module):
    """Model that performs intent and token classification
    """

    def __init__(
        self,
        num_token_labels: int,
        num_sequence_labels: int
    ) -> None:
        super().__init__()
        self.num_token_labels = num_token_labels
        self.num_sequence_labels = num_sequence_labels
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.token_classifier = torch.nn.Linear(
            768, self.num_token_labels)
        self.sequence_classifier = torch.nn.Linear(
            768,
            self.num_sequence_labels
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        token_labels=None,
        sequence_labels=None
    ) -> None:
        """Predicts the intent and token tags for a given input sequence.

        Args:
            input_ids (optional): tokenized sentence. Defaults to None.
            attention_mask (optional): attention mask to use. Defaults to None.
            token_type_ids (optional): token ids. Defaults to None.
            position_ids (optional): position ids. Defaults to None.
            head_mask (optional): head mask. Defaults to None.
            output_attentions (optional): whether to output attentions.
                Defaults to None.
            output_hidden_states (optional): whether to output hidden states.
                Defaults to None.
            token_labels (optional): true tag labels for each token to compute
                loss. Defaults to None.
            sequence_labels (optional): true class label to compute loss.
                Defaults to None.

        Returns:
            logits_token, token_loss, logits_sequence, sequence_loss
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        token_output = outputs[0]

        sequence_output = outputs[1]

        logits_token = self.token_classifier(token_output)
        logits_sequence = self.sequence_classifier(sequence_output)

        token_loss = 0
        if token_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            token_loss = loss_fct(
                logits_token.view(-1, self.num_token_labels),
                token_labels.view(-1)
            )

        sequence_loss = 0
        if sequence_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            sequence_loss = loss_fct(
                logits_sequence.view(-1, self.num_sequence_labels),
                sequence_labels.view(-1)
            )

        if token_labels is not None and sequence_labels is not None:
            return logits_token, token_loss, logits_sequence, sequence_loss
        return logits_token, logits_sequence
