import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def loss_func(output, target):
    loss = None
    return loss


class EEGtoReport(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_encoder = EEGEncoder()
        self.pos_encoder = PositionalEncoding()
        self.eeg_transformer = EEGTransformer()

    def forward(self, input, target, len, position_indicator):
        eeg_embedding = self.eeg_encoder(input)
        eeg_embedding = self.pos_encoder(eeg_embedding, position_indicator)
        word_embedding = self.eeg_transformer(eeg_embedding, target, len)
        return word_embedding

class EEGEncoder(nn.Module):
    """TO DO: encode eeg recording to embedding

    Check video at 3:17 for 1D covolutions https://www.youtube.com/watch?v=wNBaNhvL4pg
    1D convolutions

    Input:
    batch of eeg recording torch.tensor(*, 18, frequency*60)
    where '*' is sum of sample length in the batch
    E.g. torch.tensor([[[7, 0, 7, 7],
                        [0, 7, 0, 0]],

                       [[7, 0, 7, 0],
                        [0, 7, 0, 7]]]),

                        [[1, 0, 1, 1],
                        [0, 1, 0, 0]],

                       [[1, 0, 1, 0],
                        [0, 1, 0, 1]],

                        [[1, 0, 1, 0],
                         [0, 1, 0, 1]]])
                    )
            '*' = 2+3, 2 samples in this batch with length of 2 & 3 respectively

    Return:
    torch.tensor(*, 512)
    epoch of eeg recording (18, frequency*60) -> eeg embedding (512)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 512, (18,5))
        self.pool = nn.MaxPool1d(60*250-4, stride=3)
        self.batch1 = nn.BatchNorm1d(512)

    def forward(self, input):
        x = self.conv1(input).squeeze()
        x = self.pool(x).squeeze()
        eeg_embedding = self.batch1(x)
        return eeg_embedding


class PositionalEncoding(nn.Module):
    #def __init__(self):
    #    pass
    #def forward(self, eeg_embedding, len):
    #    pass
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, position_indicator):
        x = x + self.pe[position_indicator, :].squeeze()
        return self.dropout(x)


class EEGTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer()


    def forward(self, input, target, len):
        """E.g. input, target, len:"""
        input = torch.randn(5, 512)
        target = (torch.tensor([[0,0,1,0], [1,0,0,0], [0,1,0,0]]), torch.tensor([[0,1,0,0], [0,0,1,0]]))
        len = (2, 3)
        word_embedding = None
        return word_embedding
