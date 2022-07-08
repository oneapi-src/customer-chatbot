"""
Custom Handler for Torch-Serve
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

import torch
from ts.torch_handler.base_handler import BaseHandler


class ModelHandler(BaseHandler):
    """Custom handler for Torchserve testing
    """

    def preprocess(self, data):
        """Preprocess the data.

        Args:
            data: list of tokenized sentences
        """
        ids = []
        masks = []
        for sentence in data:
            token_id = torch.as_tensor(sentence['body']['ids'])
            mask = torch.as_tensor(sentence['body']['mask'])
            ids.append(token_id)
            masks.append(mask)
        return torch.stack(ids), torch.stack(masks)

    def inference(self, data, *args, **kwargs):
        """Obtain predictions from the given data.

        Args:
            data: tokenized sentence
        """
        with torch.no_grad():
            model_output = self.model.forward(data[0], data[1])
        return model_output

    def postprocess(self, data):
        """Process predictions before returning.

        Args:
            data: predictions
        """
        out = []
        for i in range(len(data[0])):
            pred = {'logit_tokens':
                    torch.argmax(data[0][i], axis=1).tolist(),
                    'logits_seq':
                        torch.argmax(data[1][i]).item()}
            out.append(pred)
        return out
