import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def loss_func(output, target):
    loss = None
    return loss


class EEGtoReport(nn.Module):
    def __init__(self, emb_dim = 512, max_len = 3):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.eeg_encoder = EEGEncoder(emb_dim = self.emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim = self.emb_dim)
        #self.eeg_transformer = EEGTransformer()

    def forward(self, input, target, length):
        eeg_embedding = self.eeg_encoder(input)
        eeg_embedding = self.pos_encoder(eeg_embedding, self.max_len, length)
        #word_embedding = self.eeg_transformer(eeg_embedding, target, len)
        #return word_embedding
        return eeg_embedding


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

    def __init__(self, emb_dim = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, (18, 5))
        self.pool1 = nn.MaxPool1d(16)
        self.batch1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(1, 64, (32, 5))
        self.pool2 = nn.MaxPool1d(16)
        self.batch2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(1, 256, (64, 5))
        self.pool3 = nn.MaxPool1d(8)
        self.batch3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(1, emb_dim, (256, 5))
        self.pool4 = nn.MaxPool1d(2)
        self.batch4 = nn.BatchNorm1d(emb_dim)

    def forward(self, input):
        input = input.unsqueeze(1)
        x = self.conv1(input).squeeze()
        x = self.pool1(x)
        x = self.batch1(x).unsqueeze(1)
        x = self.conv2(x).squeeze()
        x = self.pool2(x)
        x = self.batch2(x).unsqueeze(1)
        x = self.conv3(x).squeeze()
        x = self.pool3(x)
        x = self.batch3(x).unsqueeze(1)
        x = self.conv4(x).squeeze()
        x = self.pool4(x)
        x = self.batch4(x).squeeze()
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim=512, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb_dim = emb_dim

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, max_len, length):
        """Return: torch.tensor((S,N,E)), where S is the sequence length, N is the batch size, E is the feature number
               E.g. eeg_embedding, len = torch.tensor(5, 512), (1,4) -> torch.tensor(4, 2, 512) padding with 0 and then do positional encoding
               torch.split may be helpful
        """
        position_indicator = sum([list(range(x)) for x in length], [])
        x = x + self.pe[position_indicator, :].squeeze()
        x = torch.split(x, length)
        batch_size = len(length)
        final_embedding = torch.zeros(batch_size, max_len, self.emb_dim)
        for i, k in enumerate(length):
            final_embedding[i, :k, :] = x[i]
        return self.dropout(final_embedding)


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