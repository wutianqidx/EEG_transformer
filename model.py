import torch
import torch.nn as nn
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

    def forward(self, input, target, len):
        eeg_embedding = self.eeg_encoder(input)
        eeg_embedding = self.pos_encoder(eeg_embedding, len)
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
        self.pool = nn.MaxPool1d(5, stride=3)
        self.batch1 = nn.BatchNorm1d(512)

    def forward(self, input):
        eeg_embedding = None
        return eeg_embedding
        
class PositionalEncoding(nn.Module):
    def __init__(self):
        pass
    def forward(self, eeg_embedding, len):
        pass

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
