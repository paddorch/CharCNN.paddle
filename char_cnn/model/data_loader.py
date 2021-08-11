#!/usr/bin/env python3
# from torch.utils.data import DataLoader, Dataset
# import torch.autograd as autograd
# import torch
import json
import csv
import numpy as np

import paddle
import nlpaug.augmenter.word as naw
from paddle.io import Dataset, DataLoader
import random


class AGNEWs(Dataset):
    def __init__(self, label_data_path, alphabet_path, l0=1014, data_augment=False):
        """Create AG's News dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            alphabet_path: The path of alphabet json file.
        """
        self.label_data_path = label_data_path
        self.l0 = l0
        # read alphabet
        self.loadAlphabet(alphabet_path)
        self.load(label_data_path)
        self.data_augment = data_augment
        if self.data_augment:
        #     import nltk
        #     nltk.download('wordnet')
        #     nltk.download('averaged_perceptron_tagger')
            self.aug = naw.SynonymAug(aug_src='wordnet')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx] - 1
        return X, y

    def loadAlphabet(self, alphabet_path):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def load(self, label_data_path, lowercase=True):
        self.label = []
        self.data = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                self.label.append(int(row[0]))
                txt = ' '.join(row[1:])
                if lowercase:
                    txt = txt.lower()
                self.data.append(txt[:self.l0])  # max length

        self.y = self.label

    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = np.zeros((len(self.alphabet), self.l0), dtype="float32")
        sequence = self.data[idx]
        if self.data_augment:
            sequence = self.dataAugment(sequence)
        for index_char, char in enumerate(sequence[::-1]):  # reverse
            if self.char2Index(char) != -1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def getClassWeight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples / float(self.label.count(c)) for c in label_set]
        return class_weight, num_class

    def dataAugment(self, text):
        if random.random() > 0.5:  # synonyms replace with p = 0.5
            text = self.aug.augment(text)[:self.l0]  # can't exceed l0
        return text


def _test():
    label_data_path = '../data/ag_news_csv/test.csv'
    alphabet_path = '../config/alphabet.json'

    train_dataset = AGNEWs(label_data_path, alphabet_path, data_augment=True)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, drop_last=False)

    # size = 0
    for i_batch, sample_batched in enumerate(train_loader):
        if i_batch == 0:
            print(sample_batched)
            print(sample_batched[0][0][0].shape)

        # print(sample_batched)
        # len(i_batch)
        # print(sample_batched['label'].size())
        # inputs = sample_batched['data']
        # print(inputs.size())
        # print('type(target): ', target)

def _test_data_augment(text):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text)

if __name__ == '__main__':
    _test_data_augment("The quick brown fox jumps over the lazy dog .")
    _test_data_augment("The quick brown fox jumps over the lazy dog .")
    # _test()