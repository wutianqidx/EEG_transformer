import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def loss_func(output, target):
    loss = None
    return loss


class EEGtoReport(nn.Module):
    def __init__(self, emb_dim=512, emb_dim_t=512, eeg_epoch_max=50, report_epoch_max=174, vocab_size=78):
        super().__init__()
        self.eeg_epoch_max = eeg_epoch_max
        self.report_epoch_max = report_epoch_max
        # input eeg embedding
        self.eeg_encoder = EEGEncoder(emb_dim=emb_dim)
        # target report embedding
        self.embedding = nn.Embedding(vocab_size, emb_dim_t, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        # position encoder for src & target
        self.eeg_pos_encoder = PositionalEncoding(emb_dim=emb_dim, input_type='eeg', eeg_max=eeg_epoch_max)
        self.report_pos_encoder = PositionalEncoding(emb_dim=emb_dim_t, input_type='report')
        # transformer
        self.eeg_transformer = nn.Transformer(d_model=emb_dim, nhead=1, num_encoder_layers=1,
                                              num_decoder_layers=1, dim_feedforward=128, dropout=0.1)

    def forward(self, input, target, length, length_t):
        eeg_embedding = self.eeg_encoder(input)
        eeg_embedding, src_padding_mask = self.eeg_pos_encoder(eeg_embedding, length)

        target, tgt_padding_mask = pad_target(target, self.report_epoch_max)
        report_embedding = self.embedding(target)
        report_embedding = self.report_pos_encoder(report_embedding, length_t)

        word_embedding = self.eeg_transformer(eeg_embedding, report_embedding,
                                              src_key_padding_mask=src_padding_mask,
                                              tgt_key_padding_mask=tgt_padding_mask,
                                              memory_key_padding_mask=src_padding_mask)
        return word_embedding


def pad_target(report, max_len):
    n = len(report)
    tgt, tgt_mask = torch.zeros((n, max_len), dtype = int), torch.zeros((n, max_len), dtype = bool)
    for i, x in enumerate(report):
        tgt[i, :len(x)] = x
        tgt_mask[i, len(x):] = True
    return tgt, tgt_mask

class EEGEncoder(nn.Module):
    """BY TIANQI: encode eeg recording to embedding
    Check video at 3:17 for 1D covolutions https://www.youtube.com/watch?v=wNBaNhvL4pg
    1:15:00 https://www.youtube.com/watch?v=FrKWiRv254g
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
        self.conv1 = nn.Conv1d(18, 32, 5)
        self.pool1 = nn.MaxPool1d(23)
        self.batch1 = nn.BatchNorm1d(32)
        #self.dropout1 = nn.Dropout(0.15)
        self.conv2 = nn.Conv1d(32, 64, 5)
        self.pool2 = nn.MaxPool1d(9)
        self.batch2 = nn.BatchNorm1d(64)
        #self.dropout2 = nn.Dropout(0.15)
        self.conv3 = nn.Conv1d(64, 256, 5)
        self.pool3 = nn.MaxPool1d(4)
        self.batch3 = nn.BatchNorm1d(256)
        #self.dropout3 = nn.Dropout(0.15)
        self.conv4 = nn.Conv1d(256, emb_dim, 5)
        self.pool4 = nn.MaxPool1d(13)
        self.batch4 = nn.BatchNorm1d(emb_dim)

    def forward(self, input):
        input = input.float()
        x = self.conv1(input)
        x = self.pool1(x)
        x = F.relu(self.batch1(x))
        #x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(self.batch2(x))
        #x = self.dropout2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(self.batch3(x))
        #x = self.dropout3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = F.relu(self.batch4(x))
        return x.squeeze()


class PositionalEncoding(nn.Module):
    "BY TIANQI"
    def __init__(self, emb_dim=512, dropout=0.1, pos_max_len=5000, input_type='eeg', eeg_max=20):   #input_type=['eeg','report']
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb_dim = emb_dim
        self.input_type = input_type
        self.eeg_max = eeg_max

        pe = torch.zeros(pos_max_len, emb_dim)
        position = torch.arange(0, pos_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, length):
        """Return: torch.tensor((N,S,E)), where S is the sequence length, N is the batch size, E is the feature number
               E.g. eeg_embedding, len = torch.tensor(5, 512), (1,4) -> torch.tensor(4, 2, 512) padding with 0 and then do positional encoding
               torch.split may be helpful
        """
        if self.input_type == 'report':
            final_embedding = x + self.pe[:x.size(0), :]    # N,S,E
            return self.dropout(final_embedding.permute(1,0,2))
        elif self.input_type == 'eeg':
            position_indicator = sum([list(range(x)) for x in length], [])
            #print(x.size(), self.pe[position_indicator, :].squeeze().size())
            x = x + self.pe[position_indicator, :].squeeze()
            x = torch.split(x, length)
            batch_size = len(length)
            final_embedding = torch.zeros(batch_size, self.eeg_max, self.emb_dim)
            eeg_mask = torch.zeros((batch_size, self.eeg_max), dtype=bool)
            for i, k in enumerate(length):
                final_embedding[i, :k, :] = x[i]   # N,S,E
                eeg_mask[i, k:] = 1
            final_embedding = final_embedding.permute(1,0,2)        # N,S,E ->  S,N,E
            return self.dropout(final_embedding), eeg_mask


'''
class EEGTransformer(nn.Module):
    def __init__(self, ntoken, ninp = 512, nhead = 8, nlayers = 6):
        super().__init__()
        #encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.model_type = 'Transformer'
        self.padding_mask = None
        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=ninp, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)

    #def forward(self, input, target, len):
        #"""E.g. input, target, len:"""
        # input = torch.randn(4, 2, 512) # (S,N,E)
        # target = (torch.tensor([[0,0,1,0], [1,0,0,0], [0,1,0,0]]), torch.tensor([[0,1,0,0], [0,0,1,0]]))
        # len = (1, 4)
        # word_embedding = None
        # return word_embedding

    def forward(self, src, target, len):
        device = src.device
        S = src.size()[0]
        T = target.size()[0]
        padding_mask = self._generate_src_padding_mask(len, S).to(device)
        self.padding_mask = padding_mask

        memory = self.transformer_encoder(src, src_key_padding_mask=self.padding_mask)
        # word_embedding = self.transformer_decoder(target, memory, tgt_mask = self._generate_src_padding_mask(T))
        # return word_embedding

    def _generate_src_padding_mask(self, len, S):
        mask = []
        for i in len:
            a = i * [False] + (S-i) * [True]
            mask.append(a)
        mask = torch.tensor(mask)
        return mask

    def _generate_square_subsequent_mask(self, S):
        mask = (torch.triu(torch.ones(S, S)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

'''