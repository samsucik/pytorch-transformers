import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation.base import init_embedding, fetch_embedding


class BiRNNModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        fc_size = config.fc_size
        init_embedding(self, config)

        self.bi_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, 1, batch_first=True, bidirectional=True)        
        self.fc1 = nn.Linear(2 * self.hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, config.n_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = config.mode

    def non_embedding_params(self):
        params = []
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                continue
            params.extend(p for p in m.parameters() if p.dim() == 2)
        return params

    def forward(self, sent_tuple):
        sent, sent_length = sent_tuple
        sent = fetch_embedding(self, self.mode, sent, squash=True)#.squeeze(1)
        
        sorted_len, sorted_ind = torch.sort(sent_length, descending=True)
        _, sorted_back_ind = torch.sort(sorted_ind)
        sorted_sent = sent[sorted_ind]

        try:
            packed = torch.nn.utils.rnn.pack_padded_sequence(sorted_sent, sorted_len, batch_first=True)
        except RuntimeError as e:
            sorted_len = torch.eq(sorted_len, 0).long() + sorted_len
            packed = torch.nn.utils.rnn.pack_padded_sequence(sorted_sent, sorted_len, batch_first=True)
        rnn_seq, rnn_out = self.bi_rnn(packed)
        rnn_out = rnn_out[0].permute(1, 0, 2)[sorted_back_ind]
        rnn_out = rnn_out.contiguous().view(rnn_out.size()[0], rnn_out.size()[1] * rnn_out.size()[2])
        x = F.relu(self.fc1(rnn_out))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits
