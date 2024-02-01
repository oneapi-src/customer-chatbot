# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Code adopted from
https://github.com/sz128/slot_filling_and_intent_detection_of_SLU
"""

import operator
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def construct_vocab(
    input_seqs: List[str],
    vocab_config: Dict[str, Any] = None
) -> Union[Dict[str, int], Dict[int, str]]:
    """Construct a vocabulary given a list of sentences.

    Args:
        input_seqs (List[str]): list of sentences
        vocab_config (Dict[str, Any], optional): options for constructing
            the vocab. Defaults to None.

    Returns:
        Union[Dict[str,int], Dict[int, str]]: dictionarys for lookup and
            reverse lookup
    """

    if vocab_config is None:
        vocab_config = {'mini_word_freq': 1, 'bos_eos': False}

    vocab = {}
    for seq in input_seqs:
        if isinstance(seq, type([])):
            for word in seq:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
        else:
            if seq not in vocab:
                vocab[seq] = 1
            else:
                vocab[seq] += 1

    # Discard start, end, pad and unk tokens if already present
    if '<s>' in vocab:
        del vocab['<s>']
    if '<pad>' in vocab:
        del vocab['<pad>']
    if '</s>' in vocab:
        del vocab['</s>']
    if '<unk>' in vocab:
        del vocab['<unk>']

    if vocab_config['bos_eos'] is True:
        word2id = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        id2word = {0: '<pad>', 1: '<unk>', 2: '<s>', 3: '</s>'}
    else:
        word2id = {'<pad>': 0, '<unk>': 1, }
        id2word = {0: '<pad>', 1: '<unk>', }

    sorted_word2id = sorted(
        vocab.items(),
        key=operator.itemgetter(1),
        reverse=True
    )

    sorted_words = [x[0] for x in sorted_word2id if x[1]
                    >= vocab_config['mini_word_freq']]

    for word in sorted_words:
        idx = len(word2id)
        word2id[word] = idx
        id2word[idx] = word

    return word2id, id2word


def read_vocab_file(
        vocab_path: str,
        bos_eos: bool = False,
        no_pad: bool = False,
        no_unk: bool = False,
        separator: str = ':'
) -> Union[Dict[str, int], Dict[int, str]]:
    """Reads a pre-existing vocabulary.

    Args:
        vocab_path (str): path to vocab file
        bos_eos (bool, optional): add begining and ending. Defaults to False.
        no_pad (bool, optional): use pad tokens. Defaults to False.
        no_unk (bool, optional): use unknown tokens. Defaults to False.
        separator (str, optional): separator token  to use. Defaults to ':'.

    Returns:
        Union[Dict[str,int], Dict[int,str]]: dictionarys for lookup and
            reverse lookup
    """

    word2id, id2word = {}, {}
    if not no_pad:
        word2id['<pad>'] = len(word2id)
        id2word[len(id2word)] = '<pad>'
    if not no_unk:
        word2id['<unk>'] = len(word2id)
        id2word[len(id2word)] = '<unk>'
    if bos_eos is True:
        word2id['<s>'] = len(word2id)
        id2word[len(id2word)] = '<s>'
        word2id['</s>'] = len(word2id)
        id2word[len(id2word)] = '</s>'
    with open(vocab_path, 'r', encoding="utf8") as file:
        for line in file:
            if separator in line:
                word, idx = line.strip('\r\n').split(' '+separator+' ')
                idx = int(idx)
            else:
                word = line.strip()
                idx = len(word2id)
            if word not in word2id:
                word2id[word] = idx
                id2word[idx] = word
    return word2id, id2word


def read_vocab_from_data_file(
    data_path: str,
    vocab_config: Dict[str, Any] = None,
    with_tag: bool = True,
    separator: str = ':'
) -> Union[Dict[str, int], Dict[int, str]]:
    """Build a vocab from a data file

    Args:
        data_path (str): file path of data
        vocab_config (Dict[str, Any], optional): vocab config. Defaults to None.
        with_tag (bool, optional): use tags. Defaults to True.
        separator (_type_, optional): separator token to use. Defaults to ':'.

    Returns:
        Union[Dict[str, int], Dict[int, str]]: dictionarys for lookup and
            reverse lookup
    """

    if vocab_config is None:
        vocab_config = {'mini_word_freq': 1,
                        'bos_eos': False, 'lowercase': False}
    print('Reading source data ...')
    input_seqs = []
    with open(data_path, 'r', encoding="utf8") as file:
        for _, line in enumerate(file):
            slot_tag_line = line.strip('\n\r').split(' <=> ')[0]
            if slot_tag_line == "":
                continue
            in_seq = []
            for item in slot_tag_line.split(' '):
                if with_tag:
                    tmp = item.split(separator)
                    word, _ = separator.join(tmp[:-1]), tmp[-1]
                else:
                    word = item
                if vocab_config['lowercase']:
                    word = word.lower()
                in_seq.append(word)
            input_seqs.append(in_seq)

    print('Constructing input vocabulary from ', data_path, ' ...')
    word2idx, idx2word = construct_vocab(input_seqs, vocab_config)
    return (word2idx, idx2word)


def read_seqtag_data_with_class(
    data_path: str,
    word2idx: Dict[str, int],
    tag2idx: Dict[str, int],
    class2idx: Dict[str, int],
    separator: str = ':',
    multi_class: bool = False,
    keep_order: bool = False,
    lowercase: bool = False
) -> Union[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Read data from files.

    Args:
        data_path (str): file path of data
        word2idx (Dict[str, int]): input vocab
        tag2idx (Dict[str, int]): tag vocab
        class2idx (Dict[str, int]): classification vocab
        separator (_type_, optional): separator to use. Defaults to ':'.
        multi_class (bool, optional): multiple classifiers. Defaults to False.
        keep_order (bool, optional): keep a track of line number.
            Defaults to False.
        lowercase (bool, optional): use lowercase. Defaults to False.

    Returns:
        Union[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: input features,
            tag labels, class labels
    """

    print('Reading source data ...')
    input_seqs = []
    tag_seqs = []
    class_labels = []
    line_num = -1
    with open(data_path, 'r', encoding="utf8") as file:
        for _, line in enumerate(file):
            line_num += 1
            slot_tag_line, class_name = line.strip('\n\r').split(' <=> ')
            if slot_tag_line == "":
                continue
            in_seq, tag_seq = [], []
            for item in slot_tag_line.split(' '):
                tmp = item.split(separator)
                word, tag = separator.join(tmp[:-1]), tmp[-1]
                if lowercase:
                    word = word.lower()
                in_seq.append(
                    word2idx[word] if word in word2idx else word2idx['<unk>'])
                tag_seq.append(tag2idx[tag] if tag in tag2idx else (
                    tag2idx['<unk>'], tag))
            if keep_order:
                in_seq.append(line_num)
            input_seqs.append(in_seq)
            tag_seqs.append(tag_seq)
            if multi_class:
                if class_name == '':
                    class_labels.append([])
                else:
                    class_labels.append([class2idx[val]
                                        for val in class_name.split(';')])
            else:
                if ';' not in class_name:
                    class_labels.append(class2idx[class_name])
                else:
                    # get the first class for training
                    class_labels.append(
                        (
                            class2idx[class_name.split(';')[0]],
                            class_name.split(';')
                        )
                    )

    input_feats = {'data': input_seqs}
    tag_labels = {'data': tag_seqs}
    class_labels = {'data': class_labels}

    return input_feats, tag_labels, class_labels


class ATISDataset(Dataset):
    """Dataset for use within PyTorch
    """

    def __init__(
            self, sentences, tags, class_labels, tokenizer, max_length,
            word2id, id2word,
            class2id, id2class,
            tag2id, id2tag):

        self.len = len(sentences)
        self.sentences = sentences
        self.tags = tags
        self.class_labels = class_labels
        self.tokenizer = tokenizer
        self.max_len = max_length

        self.word2id, self.id2word = word2id, id2word
        self.class2id, self.id2class = class2id, id2class
        self.tag2id, self.id2word = tag2id, id2tag

    def __getitem__(self, index):

        sentence = self.sentences[index].strip().split()
        word_labels = self.tags[index]
        class_label = self.class2id[self.class_labels[index]]

        labels = [self.tag2id[label] for label in word_labels]

        encoding = self.tokenizer(sentence,
                                  return_offsets_mapping=True,
                                  is_split_into_words=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len
                                  )

        encoded_labels = np.ones(
            len(encoding['offset_mapping']), dtype=int) * -100

        i = 0
        for idx, mapping in enumerate(encoding['offset_mapping']):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = labels[i]
                i += 1

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels, dtype=torch.long)
        item['class_label'] = torch.as_tensor(class_label, dtype=torch.long)

        return item

    def __len__(self):
        return self.len


def load_dataset(data_path, tokenizer, max_length):
    """load the dataset

    Args:
        data_path (str): _description_
        tokenizer : transformers tokenizer_
        max_length (int): max padding length
    Returns:
        Dict[str, Any] : collection of datasets
    """
    word2id, id2word = read_vocab_from_data_file(data_path + "/train")
    class2id, id2class = read_vocab_file(data_path + "/vocab.intent")
    tag2id, id2tag = read_vocab_file(data_path + "/vocab.slot")

    def get_ds(file_name):
        input_feats, tag_labels, class_labels = read_seqtag_data_with_class(
            data_path + "/" + file_name, word2id, tag2id, class2id)
        sentences = []
        labels = []
        cls_labels = []
        for i in range(len(input_feats['data'])):
            sent = input_feats['data'][i]
            tag = tag_labels['data'][i]
            class_label = class_labels['data'][i]
            if not isinstance(class_label, int):
                class_label = class_label[0]

            sentences.append(" ".join([id2word[idx] for idx in sent]))
            labels.append([id2tag[idx] for idx in tag])
            cls_labels.append(id2class[class_label])
        return ATISDataset(sentences, labels, cls_labels,
                           tokenizer, max_length,
                           word2id, id2word,
                           class2id, id2class,
                           tag2id, id2tag)

    return {"train": get_ds("train_all"),
            "test": get_ds("test"),
            "word2id": word2id, "id2word": id2word,
            "tag2id": word2id, "id2tag": id2tag,
            "class2id": word2id, "id2class": id2class}
