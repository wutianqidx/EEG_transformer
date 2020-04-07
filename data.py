import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class EEGDataset(Dataset):
    def __init__(self, pkl_file):
        """BY PEIYAO: Load and prepare data for NN
        Args:
        pkl_file: [(eeg, report), ...], word_bag, frequency = pd.read_pickle(pkl_file)
            eeg: np.array(18, SampleLength)
            report: ['IMPRESSION', 'DESCRIPTION OF THE RECORD']
            word_bag: {word: frequency}
            frequency: n Hz which means n*60 samples/min
        """
        self.THRESHOLD = 2
        self.data, self.word_bag, self.freq = pd.read_pickle(pkl_file)
        self.textual_ids, self.ixtoword, self.wordtoix = self.build_dict()

    def build_dict(self):
        ixtoword = {1:'<end>', 2:'<sep>', 3:'<punc>', 4:'<unk>'}
        wordtoix = {'<end>':1, '<sep>':2, '<punc>':3, '<unk>':4}
        textual_ids = []

        idx = 5
        for word, freq in self.word_bag.items():
            if word not in wordtoix:
                if freq >= self.THRESHOLD:
                    ixtoword[idx] = word
                    wordtoix[word] = idx
                    idx += 1

        for eeg, report in self.data:
            temp = []
            for word in report:
                if word in wordtoix:
                    temp.append(wordtoix[word])
                else:
                    temp.append(wordtoix['<unk>'])
            textual_ids.append(temp)
        return textual_ids, ixtoword, wordtoix

    def get_text(self, idx):
        #text = torch.tensor(self.textual_ids[idx]).view(-1, 1)
        text = torch.tensor(self.textual_ids[idx])
        return text

    def get_eeg(self, idx):
        eeg, report = self.data[idx]
        shape1 = eeg.shape[0]
        shape2 = self.freq * 60
        shape0 = int(eeg.shape[1]/self.freq/60)
        cutoff = shape2 * shape0
        new_eeg = eeg[:, :cutoff]
        reshape_eeg = new_eeg.reshape(shape0, shape1, shape2)
        return torch.tensor(reshape_eeg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """BY PEIYAO: a) represent eeg from np.array(18, SampleLength) to torch.tensor(*, 18, frequency*60)
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
        eeg = self.get_eeg(idx)
        report = self.get_text(idx)
        return eeg, report, len(eeg), len(report)

def collate_wrapper(batch):
    input_indv, target_indv, length, length_t = list(zip(*batch))
    input = torch.cat(input_indv, 0)
    #target = pad_sequence(target_indv)
    target = target_indv
    return input, target, length, length_t