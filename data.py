import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class EEGDataset(Dataset):
    def __init__(self, pkl_file):
        """Load and prepare data for NN

        Args:
        pkl_file: [(eeg, report), ...], word_bag, frequency = pd.read_pickle(pkl_file)
            eeg: np.array(18, SampleLength)
            report: ['IMPRESSION', 'DESCRIPTION OF THE RECORD']
            word_bag: {word: frequency}
            frequency: n Hz which means n*60 samples/min
        """
        self.THRESHOLD = 2
        self.data, self.word_bag, self.freq = pd.read_pickle(pkl_file)
        self.textual_ids, self.MAX_LEN, self.ixtoword, self.wordtoix = self.build_dict()
        # print("self.data", self.data)
        # print("self.word_bag", self.word_bag)
        # print("self.freq", self.freq)
        """E.g. [(eeg, report), ...]:"""
        # self.data = [(np.array([[1,0,1,0],[0,1,0,1]]), ['2 0 2', '0 2 0 2 0']), (np.array([[3,0,3,0,3,0],[0,3,0,3,0,3]]), ['4 0 4', '0 4 0 4']),
        #                     (np.array([[5,0,5],[0,5,0]]), ['6 0 6 0', '0 6 0']), (np.array([[7,0,7,0,7],[0,7,0,7,0]]), ['8 0 8 0', '0 8 0 8']),
        #                     (np.array([[9,0,9,0],[0,9,0,9]]), ['0 0 0', '0 0 0 0 0 0'])]

        # self.data = [(torch.tensor([ [[1,0],[0,1]], [[1,0],[0,1]] ]), ['2 0 2', '0 2 0 2 0']), (torch.tensor([ [[3,0],[0,3]], [[3,0],[0,3]], [[3,0],[0,3]] ]), ['4 0 4', '0 4 0 4']),
        #                     (torch.tensor([ [[5,0],[0,5]] ]), ['6 0 6 0', '0 6 0']), (torch.tensor([ [[7,0],[0,7]], [[7,0],[0,7]] ]), ['8 0 8 0', '0 8 0 8']),
        #                     (torch.tensor([ [[9,0],[0,9]], [[9,0],[0,9]] ]), ['0 0 0', '0 0 0 0 0 0'])]

    def build_dict(self):
        ixtoword = {0:'<end>', 1:'<sep>', 2:'<punc>', 3:'<unk>'}
        wordtoix = {'<end>':0, '<sep>':1, '<punc>':2, '<unk>':3}
        textual_ids = []
        max_len = 0

        idx = 4
        for word, freq in self.word_bag.items():
            if word not in wordtoix:
                if freq >= self.THRESHOLD:
                    ixtoword[idx] = word
                    wordtoix[word] = idx
                    idx += 1

        for eeg, report in self.data:
            # print(eeg.shape)
            if len(report) > max_len:
                max_len = len(report)
            temp = []
            for word in report:
                if word in wordtoix:
                    temp.append(wordtoix[word])
                else:
                    temp.append(3) #<unk>
            # print(len(report))
            # print(len(temp))
            # print(report)
            # print(temp)
            textual_ids.append(temp)

        # print()
        # print(ixtoword)
        # print()
        # print(wordtoix)
        # print(self.word_bag)
        # print(max_len)
        # print(textual_ids)
        # print(self.freq)
        return textual_ids, max_len, ixtoword, wordtoix

    def get_text(self, idx):
        text = self.textual_ids[idx]
        while len(text) < self.MAX_LEN:
            text.append(0) #<end>
        return text

    def get_eeg(self, idx):
        eeg, report = self.data[idx]
        shape1 = eeg.shape[0]
        shape2 = self.freq * 60
        shape0 = int(eeg.shape[1]/self.freq/60)
        cutoff = shape2 * shape0
        new_eeg = eeg[:, :cutoff]
        reshape_eeg = new_eeg.reshape(shape0, shape1, shape2)
        return reshape_eeg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """TO DO: a) represent eeg from np.array(18, SampleLength) to torch.tensor(*, 18, frequency*60)
        SampleLength is varible but frequency*60 is fixed
        np.array([[1,0,1,0],[0,1,0,1]]) ->
        torch.tensor([ [[1,0],
                        [0,1]],

                        [[1,0],
                        [0,1]] ])
        b) represent report as one hot vector according to word bag
            ['2 0 2', '0 2 0 2 0'] -> torch.tensor([[0, 1, 0, ...], ...]) seperated by
            special token [SEP] between 'IMPRESSION' and 'DESCRIPTION OF THE RECORD'

        Args:
        idx: idx'th (eeg, report) from self.data
        Return:
        torch.tensor(*, 18, frequency*60), torch.tensor([[0, 1, 0, ...], ...]), len(torch.tensor(*, 18, frequency*60))
        """
        # eeg, report = self.data[idx]
        eeg = torch.tensor(self.get_eeg(idx))
        report = torch.tensor(self.get_text(idx))

        return eeg, report, len(eeg)

def collate_wrapper(batch):
    input_indv, target, len = list(zip(*batch))
    input = torch.cat(input_indv, 0)
    target = torch.stack(target)
    return input, target, len
